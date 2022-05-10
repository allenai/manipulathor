"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
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
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from gym.spaces.dict import Dict as SpaceDict

from utils.model_utils import LinearActorHeadNoCategory

from manipulathor_utils.debugger_util import ForkedPdb


class BringObjectBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
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
        raise Exception('Deprecated, check todos')
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        sensor_names = self.observation_space.spaces.keys()
        self.visual_encoder = SimpleCNN(
            self.observation_space,
            self._hidden_size,
            rgb_uuid="rgb_lowres" if "rgb_lowres" in sensor_names else None,
            depth_uuid="depth_lowres" if "depth_lowres" in sensor_names else None,
        )

        if "rgb_lowres" in sensor_names and "depth_lowres" in sensor_names:
            input_visual_feature_num = 2
        elif "rgb_lowres" in sensor_names:
            input_visual_feature_num = 1
        elif "depth_lowres" in sensor_names:
            input_visual_feature_num = 1

        self.state_encoder = RNNStateEncoder(
            (self._hidden_size) * input_visual_feature_num,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pickup = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_pickup = LinearCriticHead(self._hidden_size)
        self.actor_dropoff = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_dropoff = LinearCriticHead(self._hidden_size)
        # initial_dist_embedding_size = torch.Tensor([3, 100, obj_state_embedding_size])
        # self.initial_dist_embedding = input_embedding_net(
        #     initial_dist_embedding_size.long().tolist(), dropout=0
        # )


        self.train()


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

        #LATER_TODO maybe we really need to switch to resnet now that visual features are actually important

        #LATER_TODO use object type as input

        #LATER_TODO check object nav to see what they do


        perception_embed = self.visual_encoder(observations)
        x_out, rnn_hidden_states = self.state_encoder(
            perception_embed, memory.tensor("rnn"), masks
        )

        # I think we need two model one for pick up and one for drop off

        actor_out_pickup = self.actor_pickup(x_out)
        critic_out_pickup = self.critic_pickup(x_out)


        actor_out_dropoff = self.actor_dropoff(x_out)
        critic_out_dropoff = self.critic_dropoff(x_out)

        actor_out_final = actor_out_dropoff
        critic_out_final = critic_out_dropoff

        pickup_bool = observations["pickedup_object"]
        before_pickup = pickup_bool == 0  # not used because of our initialization

        actor_out_final[before_pickup] = actor_out_pickup[before_pickup]
        critic_out_final[before_pickup] = critic_out_pickup[before_pickup]

        actor_out = CategoricalDistr(logits=actor_out_final)

        #LATER_TODO check to get gradients on both of them

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)


        return (
            actor_critic_output,
            memory,
        )
