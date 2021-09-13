"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import platform

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from typing import Tuple, Optional

import gym
import torch
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from detectron2.modeling import build_model
from gym.spaces.dict import Dict as SpaceDict

from legacy.armpointnav_baselines.models import LinearActorHeadNoCategory
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.thor_category_names import thor_object_name_to_lvis_valid_indices, thor_possible_objects


class SmallBringObjectPredictBBXDepthBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model for preddistancenav task.

    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        obj_state_embedding_size=512,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type="GRU",
        teacher_forcing=1,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        print('Deprecated')
        ForkedPdb().set_trace()
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        # sensor_names = self.observation_space.spaces.keys()
        network_args = {'input_channels': 2, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)

        self.state_encoder = RNNStateEncoder(
            512,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pickup = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_pickup = LinearCriticHead(self._hidden_size)

        self.train()

        print('makign the predictor part')

        self.predictor_name = "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        self.pred_thr = 0.1 #LATER_TODO is this good value?
        self.bbox_predictor = self.initialize_bbox_predictor()
        self.bbox_predictor.eval()

        print('done with making the model')

    # def initialize_bbox_predictor(self):
    #     cfg = get_cfg()
    #     # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    #     cfg.merge_from_file(model_zoo.get_config_file(self.predictor_name))
    #     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.pred_thr#0.5  # set threshold for this model
    #     # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.predictor_name)
    #
    #     #LATER_TODO this is just for the sake of testing on mac
    #     if platform.system() == "Darwin":
    #         cfg.MODEL.DEVICE = "cpu"
    #
    #     predictor = DefaultPredictor(cfg)
    #     return predictor

    def initialize_bbox_predictor(self): #LATER_TODO seems like this is fully on gpu 0 and we need to save it maybe?
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(self.predictor_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.pred_thr#0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.predictor_name)

        #LATER_TODO remove later
        cfg.MODEL.DEVICE = "cpu"

        #LATER_TODO this is just for the sake of testing on mac
        if platform.system() == "Darwin":
            cfg.MODEL.DEVICE = "cpu"

        model = build_model(cfg)  # returns a torch.nn.Module
        file_url = 'https://dl.fbaipublicfiles.com/detectron2/LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/144219072/model_final_571f7c.pkl'
        DetectionCheckpointer(model).load(file_url)
        model.eval()
        return model


    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def generate_mask(self, classes, boxes, object_type, im_size):
        object_name = thor_possible_objects[object_type.item()]
        valid_indices = thor_object_name_to_lvis_valid_indices[object_name]
        output_mask = torch.zeros(im_size)
        for i in range(len(boxes)):
            cls = classes[i].item()
            box = boxes[i].tensor.round().int().squeeze().tolist()
            x1, y1, x2, y2 = box
            if cls in valid_indices:
                output_mask[y1:y2, x1:x2] = 1. #LATER_TODO should this be swapped?
        return output_mask.unsqueeze(-1)

    def get_bounding_boxes(self, rgb_images, object, destination): #LATER_TODO this might be super slow but it is what it is

        if not self.bbox_predictor.device == torch.device('cpu'): #LATER_TODO remove
            print('HAD TO MOVE TO CPU')
            self.bbox_predictor = self.bbox_predictor.to('cpu')

        #LATER_TODO remove
        original_device = rgb_images.device
        rgb_images = rgb_images.cpu()


        with torch.no_grad():
            bsize, seqlen, w, h, c = rgb_images.shape
            rgb_images = rgb_images.view(bsize * seqlen, w, h, c).permute(0, 3, 1, 2)[:,[2,1,0], :, :] #LATER_TODO Changin RGB to BGR
            object = object.view(bsize * seqlen,1)
            destination = destination.view(bsize * seqlen,1)
            all_object_masks = torch.zeros((bsize * seqlen, w, h, 1)).to(rgb_images.device)
            all_destination_masks = torch.zeros((bsize * seqlen, w, h, 1)).to(rgb_images.device)
            self.bbox_predictor.eval() # Why do I have to do this everytime?
            outputs = self.bbox_predictor([dict(image=rgb_images[i]) for i in range(bsize * seqlen)]) #LATER_TODO should this be raw image?
            for i in range(bsize * seqlen):
                classes = (outputs[i]["instances"].pred_classes)
                boxes = (outputs[i]["instances"].pred_boxes)
                # scores = (outputs["instances"].scores)#LATER_TODO we can later on use this to incorporate it inside this

                object_mask = self.generate_mask(classes, boxes, object[i], (w,h))
                destination_mask = self.generate_mask(classes, boxes, destination[i], (w,h))

                all_object_masks[i] = object_mask
                all_destination_masks[i] = destination_mask


            all_destination_masks = all_destination_masks.detach()
            all_object_masks = all_object_masks.detach()

            #LATER_TODO remove
            all_destination_masks = all_destination_masks.to(original_device).detach()
            all_object_masks = all_object_masks.to(original_device).detach()


        return all_object_masks.view(bsize, seqlen, w, h, 1), all_destination_masks.view(bsize, seqlen, w, h, 1)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        #we really need to switch to resnet now that visual features are actually important


        predicted_object_boxes, predicted_destination_boxes = self.get_bounding_boxes(observations['raw_rgb_lowres'], observations['target_object_type'], observations['target_location_type'])


        target_object_observation = torch.cat([observations['depth_lowres'],predicted_object_boxes], dim=-1).float()
        target_location_observation = torch.cat([observations['depth_lowres'],predicted_destination_boxes], dim=-1).float()

        pickup_bool = observations["pickedup_object"]
        after_pickup = pickup_bool == 1
        visual_observation = target_object_observation
        visual_observation[after_pickup] = target_location_observation[after_pickup]

        visual_observation_encoding = compute_cnn_output(self.full_visual_encoder, visual_observation)



        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )


        # I think we need two model one for pick up and one for drop off

        actor_out_pickup = self.actor_pickup(x_out)
        critic_out_pickup = self.critic_pickup(x_out)


        actor_out_final = actor_out_pickup
        critic_out_final = critic_out_pickup

        actor_out = CategoricalDistr(logits=actor_out_final)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)


        return (
            actor_critic_output,
            memory,
        )
