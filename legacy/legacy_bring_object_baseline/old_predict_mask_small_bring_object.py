"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import os
import platform
from datetime import datetime
from typing import Tuple, Optional

import cv2
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
from gym.spaces.dict import Dict as SpaceDict

from legacy.armpointnav_baselines.models import LinearActorHeadNoCategory
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from manipulathor_utils.debugger_util import ForkedPdb


class SmallBringObjectWPredictMaskDepthBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
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
        print('deprecated, resolve todo')
        ForkedPdb().set_trace()
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        # sensor_names = self.observation_space.spaces.keys()
        network_args = {'input_channels': 2, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)

        self.detection_model = ConditionalDetectionModel()

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
        self.detection_model.eval()

        #LATER_TODO reload the weights
        weight_dir = 'datasets/apnd-dataset/weights/full_detection_full_thor_all_objects_segmentation_resnet_2021-07-30_14:47:36_model_state_324.pytar'
        detection_weight_dict = torch.load(weight_dir, map_location='cpu')
        detection_state_dict = self.detection_model.state_dict()
        for key in detection_state_dict:
            param = detection_weight_dict[key]
            detection_state_dict[key].copy_(param)
        remained = [k for k in detection_weight_dict if k not in detection_state_dict]
        assert len(remained) == 0
        #LATER_TODO is the reload correct?

        #LATER_TODO this is really bad. Does this even work? you are the worst
        weight_dir = 'datasets/apnd-dataset/weights/exp_QueryObjGTMaskSimpleDiverseBringObject_noise_0.2__stage_00__steps_000048243775.pt'
        loaded_rl_model_weights = torch.load(weight_dir, map_location='cpu')['model_state_dict']
        rl_model_state_keys = [k for k in self.state_dict() if k.replace('detection_model.', '') not in detection_state_dict]
        #LATER_TODO this is a freaking small model!


        # print('norm', self.full_visual_encoder.conv_0.weight.norm(), 'mean', self.full_visual_encoder.conv_0.weight.mean())

        rl_model_state_dict = self.state_dict()

        ForkedPdb().set_trace()
        for key in rl_model_state_keys:
            param = loaded_rl_model_weights[key]
            rl_model_state_dict[key].copy_(param)
        # print('norm', self.full_visual_encoder.conv_0.weight.norm(), 'mean', self.full_visual_encoder.conv_0.weight.mean())

    def get_detection_masks(self, query_images, images): #LATER_TODO can we save the detections so we don't have to go through them again?
        #LATER_TODO make sure the weights have stayed the same
        self.detection_model.eval()
        with torch.no_grad():
            images = images.permute(0,1,4,2,3) #Turn wxhxc to cxwxh

            batch, seqlen, c, w, h = images.shape

            images = images.view(batch * seqlen, c, w, h)
            query_images = query_images.view(batch * seqlen, c, w, h)
            #LATER_LATER_TODO visualize the outputs
            predictions = self.detection_model(dict(rgb=images, target_cropped_object=query_images))
            probs_mask = predictions['object_mask']
            probs_mask = probs_mask.view(batch, seqlen, 2, w, h)
            mask = probs_mask.argmax(dim=2).float().unsqueeze(-1)#To add the channel back in the end of the image
            return mask


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


        query_source_objects = observations['category_object_source']
        query_destination_objects = observations['category_object_destination']


        pickup_bool = observations["pickedup_object"]
        after_pickup = pickup_bool == 1

        query_objects = query_source_objects
        query_objects[after_pickup] = query_destination_objects[after_pickup]

        predicted_masks = self.get_detection_masks(query_objects, observations['only_detection_rgb_lowres'])

        visual_observation = torch.cat([observations['depth_lowres'],predicted_masks], dim=-1).float()

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

        self.visualize = platform.system() == "Darwin"
        #LATER_TODO really bad design
        if self.visualize:
            def unnormalize_image(img):
                mean=torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
                std=torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
                img = (img * std + mean)
                img = torch.clamp(img, 0, 1)
                return img
            viz_image = observations['only_detection_rgb_lowres']
            depth = observations['depth_lowres']
            predicted_masks
            bsize, seqlen, w, h, c = viz_image.shape
            if bsize == 1 and seqlen == 1:
                viz_image = viz_image.squeeze(0).squeeze(0)
                viz_query_obj = query_objects.squeeze(0).squeeze(0).permute(1,2,0) #TO make it channel last
                viz_mask = predicted_masks.squeeze(0).squeeze(0).repeat(1,1, 3)
                viz_image = unnormalize_image(viz_image)
                viz_query_obj = unnormalize_image(viz_query_obj)
                combined = torch.cat([viz_image, viz_query_obj, viz_mask], dim=1)
                directory_to_write_images = 'experiment_output/visualizations_masks'
                os.makedirs(directory_to_write_images, exist_ok=True)
                now = datetime.now()
                time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f.png")
                cv2.imwrite(os.path.join(directory_to_write_images, time_to_write), (combined[:,:,[2,1,0]] * 255.).int().numpy())





        # memory = memory.check_append("predicted_segmentation_mask", predicted_masks.detach())
        # actor_critic_output.extras['predicted_segmentation_mask'] = predicted_masks.detach()


        return (
            actor_critic_output,
            memory,
        )
