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

from manipulathor_baselines.armpointnav_baselines.models.base_models import LinearActorHeadNoCategory
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from manipulathor_utils.debugger_util import ForkedPdb
from manipulathor_utils.net_utils import input_embedding_net


class SmallBringObjectWQueryObjGtMaskDepthBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
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
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        # sensor_names = self.observation_space.spaces.keys()
        network_args = {'input_channels': 5, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)

        # self.detection_model = ConditionalDetectionModel()

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
        # self.detection_model.eval()



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

        source_object_mask = observations['object_mask_source']
        destination_object_mask = observations['object_mask_destination']

        gt_mask = source_object_mask
        gt_mask[after_pickup] = destination_object_mask[after_pickup]

        visual_observation = torch.cat([observations['depth_lowres'],query_objects.permute(0, 1, 3, 4, 2), gt_mask], dim=-1).float()

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
        #TODO really bad design
        if self.visualize:
            def unnormalize_image(img):
                mean=torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
                std=torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
                img = (img * std + mean)
                img = torch.clamp(img, 0, 1)
                return img
            viz_image = observations['only_detection_rgb_lowres']
            depth = observations['depth_lowres']
            predicted_masks = gt_mask
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
