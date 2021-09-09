"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import copy
import os
import platform
import random
from datetime import datetime
from typing import Tuple, Optional

import cv2
import gym
import torch
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from gym.spaces.dict import Dict as SpaceDict
from torch import nn
from torchvision import models

from ithor_arm.ithor_arm_constants import DONT_USE_ALL_POSSIBLE_OBJECTS_EVER
from manipulathor_baselines.armpointnav_baselines.models.base_models import LinearActorHeadNoCategory
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from manipulathor_utils.debugger_util import ForkedPdb
from manipulathor_utils.net_utils import input_embedding_net
from utils.hacky_viz_utils import hacky_visualization, calc_dict_average


class PredictMaskSmallBringObjectWQueryObjRGBDModel(ActorCriticModel[CategoricalDistr]):
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
        visualize=False,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.visualize = visualize

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        # sensor_names = self.observation_space.spaces.keys()
        network_args = {'input_channels': 8, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
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

        self.metrics = {'mask_loss':[], 'per_category':{}, 'confusion_matrix':{}, 'object_appeared':{}, 'average_pixel_appearance': {}} #TODO can it be any worse?

        self.object_detected_per_episode = {}
        self.all_episode_names = set()

        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

        detection_weight_dir = None
        policy_weight_dir = None

        #TODO remove
        # detection_weight_dir = '/Users/kianae/Desktop/important_weights/detection_without_color_jitter_model_state_271.pytar'
        # policy_weight_dir = '/Users/kianae/Desktop/exp_SmallNoiseRGBQueryObjGTMaskSimpleDiverseBringObject_continue_training_w_noise_35__stage_00__steps_000163313525.pt'
        # # policy_weight_dir = '/Users/kianae/Desktop/important_weights/exp_NoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject_no_pu_no_noise_query_obj_w_gt_mask_and_rgb__stage_00__steps_000065308765.pt'
        # policy_weight_dir = '/Users/kianae/Desktop/exp_SmallNoiseRGBQueryObjGTMaskSimpleDiverseBringObject_continue_training_w_noise_0.2__stage_00__steps_000070844861.pt'


        #TODO reload the weights really bad design choice
        if detection_weight_dir is not None:
            detection_weight_dict = torch.load(detection_weight_dir, map_location='cpu')
            detection_state_dict = self.detection_model.state_dict()
            for key in detection_state_dict:
                param = detection_weight_dict[key]
                detection_state_dict[key].copy_(param)
            remained = [k for k in detection_weight_dict if k not in detection_state_dict]
            # assert len(remained) == 0
            print(
                'WARNING!',
                remained
            )
        else:
            print('CAREFUL! NO DETECTION WEIGHT, THIS IS USELESS')
        if policy_weight_dir is not None:
            loaded_rl_model_weights = torch.load(policy_weight_dir, map_location='cpu')['model_state_dict']
            rl_model_state_keys = [k for k in self.state_dict() if k.replace('detection_model.', '') not in detection_state_dict]
            rl_model_state_dict = self.state_dict()

            for key in rl_model_state_keys:
                param = loaded_rl_model_weights[key]
                rl_model_state_dict[key].copy_(param)

    def get_detection_masks(self, query_images, images):
        self.detection_model.eval()
        with torch.no_grad():
            images = images.permute(0,1,4,2,3) #Turn wxhxc to cxwxh

            batch, seqlen, c, w, h = images.shape

            images = images.view(batch * seqlen, c, w, h)
            query_images = query_images.view(batch * seqlen, c, w, h)
            #LATER_TODO visualize the outputs
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

        visual_observation = torch.cat([observations['depth_lowres'], observations['rgb_lowres'],query_objects.permute(0, 1, 3, 4, 2), predicted_masks], dim=-1).float()

        visual_observation_encoding = compute_cnn_output(self.full_visual_encoder, visual_observation)

        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )

        # self.visualize = platform.system() == "Darwin"
        # TODO really bad design
        if self.visualize:
            gt_mask = observations['gt_mask_for_loss_source']
            gt_mask[after_pickup] = observations['gt_mask_for_loss_destination'][after_pickup]
            hacky_visualization(observations, object_mask=predicted_masks, query_objects=query_objects, base_directory_to_right_images=self.starting_time, gt_mask = gt_mask)


        actor_out_pickup = self.actor_pickup(x_out)
        critic_out_pickup = self.critic_pickup(x_out)


        actor_out_final = actor_out_pickup
        critic_out_final = critic_out_pickup

        actor_out = CategoricalDistr(logits=actor_out_final)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)

        actor_critic_output.extras['predicted_mask'] = predicted_masks.detach()

        self.calc_losses(observations, actor_critic_output)


        return (
            actor_critic_output,
            memory,
        )
    def calc_losses(self, observations, actor_critic_output):
        pickup_bool = observations['pickedup_object'].cpu()
        after_pickup = pickup_bool == 1

        extra_model_outputs = actor_critic_output.extras
        predicted_mask = extra_model_outputs['predicted_mask'].cpu()
        gt_masks_destination = observations['gt_mask_for_loss_destination'].cpu()
        gt_masks_source = observations['gt_mask_for_loss_source'].cpu()
        gt_masks = gt_masks_source
        gt_masks[after_pickup] = gt_masks_destination[after_pickup]
        gt_masks = gt_masks.cpu()
        category_id = observations['temp_category_code_source']
        category_id[after_pickup] = observations['temp_category_code_destination'][after_pickup]
        predicted_mask = predicted_mask.cpu()
        all_masks = observations['all_masks_sensor'].cpu()

        # To make sure we only do this for the ones with seqlen and bsize one
        if pickup_bool.shape != torch.zeros((1,1)).shape:
            b_size, seq_len = pickup_bool.shape
            for i in range(b_size):
                for j in range(seq_len):

                    if 'temp_episode_number' in observations:
                        episode_number = observations['temp_episode_number'][i][j].item()
                    else:
                        episode_number = None
                    self.log_all(pickup_bool[i:i+1, j:j+1], gt_masks[i:i+1, j:j+1], predicted_mask[i:i+1, j:j+1], category_id[i:i+1, j:j+1], all_masks[i:i+1, j:j+1], episode_number)
        else:
            if 'temp_episode_number' in observations:
                episode_number = observations['temp_episode_number'].item()
            else:
                episode_number = None
            self.log_all(pickup_bool, gt_masks, predicted_mask, category_id, all_masks, episode_number)

    def log_all(self, pickup_bool, gt_masks, predicted_mask, category_id, all_masks, episode_number):
        assert pickup_bool.shape == torch.zeros((1,1)).shape
        gt_masks = gt_masks.squeeze(0).squeeze(0)
        predicted_mask = predicted_mask.squeeze(0).squeeze(0)
        all_masks = all_masks.squeeze(0).squeeze(0)
        object_name = DONT_USE_ALL_POSSIBLE_OBJECTS_EVER[int(category_id.item())]

        intersection = (gt_masks + predicted_mask) == 2
        union = (gt_masks + predicted_mask) > 0

        #The following is the proxy for calculating the amount that an object has appeard
        if gt_masks.sum() == 0:
            object_appeared = 0
        else:
            object_appeared = 1
        self.metrics['object_appeared'].setdefault(object_name, [])
        self.metrics['object_appeared'][object_name].append(object_appeared)
        self.metrics['average_pixel_appearance'].setdefault(object_name, [])
        self.metrics['average_pixel_appearance'][object_name].append(gt_masks.sum())

        if episode_number is not None:
            episode_number = f'{episode_number}_{object_name}'
            self.all_episode_names.add(episode_number)
            if intersection.sum() > 5: #
                self.object_detected_per_episode.setdefault(episode_number, 0)
                self.object_detected_per_episode[episode_number] += 1



        if union.sum() == 0: #it means no mask is provided and none was needed:
            # for the frames that the object does not exist we should not calculate any!
            return

        interaction_sum = intersection.sum()
        union_sum = union.sum()

        mean_iou = (interaction_sum / union_sum) #union sum is definitely bigger than 1
        total_loss = mean_iou.item()
        self.metrics['mask_loss'].append(total_loss)

        self.metrics['per_category'].setdefault(object_name, [])
        self.metrics['per_category'][object_name].append(total_loss)


        prediction_overlap = all_masks[predicted_mask.squeeze(-1) == 1]
        if len(prediction_overlap) > 0:
            value, indices = torch.mode(prediction_overlap)
            if len(value.shape) > 0:
                print('overlap is distributed', value, indices)
                value = value[0]
                indices = indices[0]
            if indices > len(prediction_overlap) / 2 and len(prediction_overlap) > 10: #mask size bigger than a threshold
                value = value.item()
                if value == -1:
                    mistaken_category = 'Background'
                else:
                    mistaken_category = DONT_USE_ALL_POSSIBLE_OBJECTS_EVER[int(value)]
                self.metrics['confusion_matrix'].setdefault(object_name, {})
                self.metrics['confusion_matrix'][object_name].setdefault(mistaken_category, 0)
                self.metrics['confusion_matrix'][object_name][mistaken_category] += 1


        if len(self.metrics['mask_loss']) % 100 == 0 or platform.system() == "Darwin":
            total = len(self.metrics['mask_loss'])
            print(f'for total{total} average is', calc_dict_average(self.metrics))
            print('number of episodes with at least 1 detection', len(self.object_detected_per_episode))
            print('average_number_of_positive_frames', sum([v for v in self.object_detected_per_episode.values()]) / (len(self.object_detected_per_episode) + 1e-10))
            print('set is', self.all_episode_names)
            print(self.object_detected_per_episode)