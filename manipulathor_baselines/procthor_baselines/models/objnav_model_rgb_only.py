"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import platform
from datetime import datetime
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
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from manipulathor_utils.debugger_util import ForkedPdb
from utils.model_utils import LinearActorHeadNoCategory
from utils.hacky_viz_utils import hacky_visualization


class ObjNavOnlyRGBModel(ActorCriticModel[CategoricalDistr]):
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
        network_args = {'input_channels': 3, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)

        # self.detection_model = ConditionalDetectionModel()
        # self.pointnav_embedding = nn.Sequential(
        #     nn.Linear(3, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 512),
        # )

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

        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))



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

        # pickup_bool = observations["pickedup_object"]
        # after_pickup = pickup_bool == 1

        visual_observation = observations['rgb_lowres']

        visual_observation_encoding = compute_cnn_output(self.full_visual_encoder, visual_observation)


        # arm_distance_to_obj_source = observations['arm_point_nav_source']
        # arm_distance_to_obj_destination = observations['arm_point_nav_destination']

        # arm_distance_to_obj_source_embedding = self.pointnav_embedding(arm_distance_to_obj_source)
        # arm_distance_to_obj_destination_embedding = self.pointnav_embedding(arm_distance_to_obj_destination)
        # pointnav_embedding = arm_distance_to_obj_source_embedding
        # pointnav_embedding[after_pickup] = arm_distance_to_obj_destination_embedding[after_pickup]

        #TODO remove as soon as the bug is resolved
        # assert not torch.any(torch.isinf(pointnav_embedding) + torch.isnan(pointnav_embedding)), 'pointnav_embedding is nan'
        assert not torch.any(torch.isinf(visual_observation) + torch.isnan(visual_observation)), 'visual_observation is nan'

        # arm_distance_to_obj = arm_distance_to_obj_source
        # arm_distance_to_obj[after_pickup] = arm_distance_to_obj_destination[after_pickup]
        # pointnav_embedding = self.pointnav_embedding(arm_distance_to_obj)

        visual_observation_encoding = torch.cat([visual_observation_encoding], dim=-1)


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

        #TODO remove as soon as bug is resolved
        actor_is_nan = torch.isinf(actor_out_final) + torch.isnan(actor_out_final)
        if torch.any(actor_is_nan):
            print('actor is nan', actor_is_nan.sum())
            print('scene number', observations['scene_number'])

        # # TODO really bad design
        # if self.visualize:
        #     arm_distance_to_obj_source = observations['arm_point_nav_source']
        #     arm_distance_to_obj_destination = observations['arm_point_nav_destination']
        #     distance_vector_to_viz = arm_distance_to_obj_source.clone()
        #     distance_vector_to_viz[after_pickup] = arm_distance_to_obj_destination.clone()[after_pickup]
        #     distance_vector_to_viz = dict(arm_dist=distance_vector_to_viz, agent_dist=distance_vector_to_viz)

        #     hacky_visualization(observations, base_directory_to_right_images=self.starting_time, distance_vector_to_viz=distance_vector_to_viz)


        return (
            actor_critic_output,
            memory,
        )

