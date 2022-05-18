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
    ObservationType, LinearActorHead,
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


class TwoCameraResnetObjNavActorCritic(ActorCriticModel[CategoricalDistr]):
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
        goal_dims=32,
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

        self.resnet_compressor = nn.Sequential(
                nn.Conv2d(2048, 128, 1),
                nn.ReLU(),
                nn.Conv2d(128, 32, 1),
                nn.ReLU(),
            )
        
        self.resnet_compressor_arm = nn.Sequential(
                nn.Conv2d(2048, 128, 1),
                nn.ReLU(),
                nn.Conv2d(128, 32, 1),
                nn.ReLU(),
            )

        self.state_encoder = RNNStateEncoder(
            512,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        self.action_space_length = action_space.n
        self.actor_network = LinearActorHead(self._hidden_size, action_space.n)
        self.critic_network = LinearCriticHead(self._hidden_size)

        self.target_obs_combiner = nn.Sequential(
                nn.Linear(32 * 7 * 7 * 2,1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )

        self.train()

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

        visual_embedding_front_cam = observations['rgb_clip_resnet']
        seq_len, bsize, c, w, h = visual_embedding_front_cam.shape
        visual_embedding_front_cam = visual_embedding_front_cam.view(seq_len * bsize, c, w, h)
        visual_embedding_front_cam = self.resnet_compressor(visual_embedding_front_cam)
        visual_embedding_front_cam = visual_embedding_front_cam.view(seq_len * bsize, -1)
        visual_embedding_front_cam = visual_embedding_front_cam.view(seq_len, bsize, 32 * 7 * 7)

        visual_embedding_arm_cam = observations['rgb_clip_resnet_arm']
        seq_len, bsize, c, w, h = visual_embedding_arm_cam.shape
        visual_embedding_arm_cam = visual_embedding_arm_cam.view(seq_len * bsize, c, w, h) #([1, 2048, 7, 7])
        visual_embedding_arm_cam_compressed = self.resnet_compressor_arm(visual_embedding_arm_cam) #([1, 32, 7, 7])
        visual_embedding_arm_cam_compressed = visual_embedding_arm_cam_compressed.view(seq_len * bsize, -1) #([1, 1568])
        visual_embedding_arm_cam_compressed = visual_embedding_arm_cam_compressed.view(seq_len, bsize, 32 * 7 * 7) #([1, 1, 1568])

        visual_observation_encoding = torch.cat([visual_embedding_front_cam,visual_embedding_arm_cam_compressed], dim=-1)
        visual_observation_encoding = self.target_obs_combiner(visual_observation_encoding)

        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )

        actor_out = self.actor_network(x_out)
        critic_out = self.critic_network(x_out)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)


        # TODO really bad design
        if self.visualize:
            hacky_visualization(observations, base_directory_to_right_images=self.starting_time)


        return (
            actor_critic_output,
            memory,
        )

