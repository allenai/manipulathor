"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import copy
import random
from typing import Tuple, Optional

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
from manipulathor_utils.debugger_util import ForkedPdb
from manipulathor_utils.net_utils import input_embedding_net


class BringObjectResnetWrapper(nn.Module):
    def __init__(self, pretrained=True, flatten=True):
        super().__init__()
        self.resnet_encoder = models.resnet18(pretrained=pretrained)
        del self.resnet_encoder.fc
        del self.resnet_encoder.avgpool
        self.flatten = flatten
        layer_1_weights= self.resnet_encoder.conv1.weight

        self.resnet_encoder.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        copied_weights = torch.zeros(self.resnet_encoder.conv1.weight.shape).float()
        copied_weights[:, :3, :, :] = layer_1_weights.clone()
        copied_weights[:,3:, :,:] = layer_1_weights.clone()[:, :2]
        with torch.no_grad():
            self.resnet_encoder.conv1.weight.copy_(copied_weights)
        # #TODO how about batchnorm stufF? how about bias?

        self.final_fc_layer = nn.Sequential(nn.ReLU(), nn.Conv2d(512, 64, 1, 1))

    def forward(self, input_to_cnn):

        x = self.resnet_encoder.conv1(input_to_cnn)
        x = self.resnet_encoder.bn1(x)
        x = self.resnet_encoder.relu(x)
        x = self.resnet_encoder.maxpool(x)
        x = self.resnet_encoder.layer1(x)
        x = self.resnet_encoder.layer2(x)
        x = self.resnet_encoder.layer3(x)
        x = self.resnet_encoder.layer4(x)
        x = self.final_fc_layer(x)
        if self.flatten:
            b_size, c, w, h = x.shape
            x = x.contiguous() #TODO do we have to have this?
            x = x.view(b_size, c * w * h)
        return x

class PickUpWMaskBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
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

        self.full_visual_encoder = BringObjectResnetWrapper()

        self.state_encoder = RNNStateEncoder(
            64 * 7 * 7,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pickup = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_pickup = LinearCriticHead(self._hidden_size)

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

        #we really need to switch to resnet now that visual features are actually important

        visual_observation = torch.cat([observations['rgb_lowres'], observations['depth_lowres'],observations['target_object_mask']], dim=-1 ).float()


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
