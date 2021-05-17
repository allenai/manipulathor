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
    LinearActorHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from gym.spaces.dict import Dict as SpaceDict

from manipulathor_utils.debugger_util import ForkedPdb
from manipulathor_utils.net_utils import input_embedding_net


class PredDistanceBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
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
        teacher_forcing=True,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
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
            (self._hidden_size) * input_visual_feature_num + obj_state_embedding_size,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)
        initial_dist_embedding_size = torch.Tensor([3, 100, obj_state_embedding_size])
        self.initial_dist_embedding = input_embedding_net(
            initial_dist_embedding_size.long().tolist(), dropout=0
        )

        self.create_distance_pred_model()

        self.teacher_forcing = teacher_forcing  #TODO we need to switch this eventually

        self.train()

    def create_distance_pred_model(self):
        distance_pred_size = torch.Tensor([512 * 3, 512, 100, 3])
        self.arm_distance_embedding = input_embedding_net(
            distance_pred_size.long().tolist(), dropout=0
        )
        self.object_distance_embedding = input_embedding_net(
            distance_pred_size.long().tolist(), dropout=0
        )

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

    def get_distance_embedding(self, state_tensor: torch.Tensor) -> torch.FloatTensor:

        return self.initial_dist_embedding(state_tensor)

    def predict_relative_distance(
        self, initial_arm2obj_dist, initial_obj2goal_dist, perception_embed, hidden_rnn
    ):
        #TODO we might have to increase hidden size because this is too much, it has to calc both object distance and arm distance, maybe combine them into one make it too hard, maybe two separate network?

        arm_relative_pred = self.arm_distance_embedding(
            torch.cat([initial_arm2obj_dist, perception_embed, hidden_rnn], dim=-1)
        )
        object_relative_pred = self.object_distance_embedding(
            torch.cat([initial_obj2goal_dist, perception_embed, hidden_rnn], dim=-1)
        )
        return {
            "relative_agent_arm_to_obj": arm_relative_pred,
            "relative_obj_to_goal": object_relative_pred,
        }

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
        initial_arm2obj_dist = self.get_distance_embedding(  #TODO is it okay to use the same embedding for both relative and initial?
            observations["initial_agent_arm_to_obj"]
        )
        initial_obj2goal_dist = self.get_distance_embedding(
            observations["initial_obj_to_goal"]
        )  #TODO maybe we can also input this?


        perception_embed = self.visual_encoder(observations)


        arm2obj_dist_list = []
        obj2goal_dist_list = []
        x_out_list = []
        for i in range(perception_embed.shape[0]):
            #TODO we have to debug solely predicting distance first
            prediction = self.predict_relative_distance(
                initial_arm2obj_dist[i : (i + 1)],
                initial_obj2goal_dist[i : (i + 1)],
                perception_embed[i : (i + 1)],
                memory.tensor("rnn"),
            )

            if self.teacher_forcing:
                arm2obj_dist = self.get_distance_embedding(
                    observations["relative_agent_arm_to_obj"][i : (i + 1)]
                )
                obj2goal_dist = self.get_distance_embedding(
                    observations["relative_obj_to_goal"][i : (i + 1)]
                )
            else:
                arm2obj_dist = self.get_distance_embedding(
                    prediction["relative_agent_arm_to_obj"]
                )
                obj2goal_dist = self.get_distance_embedding(
                    prediction["relative_obj_to_goal"]
                )

            arm2obj_dist_list.append(
                prediction["relative_agent_arm_to_obj"]  # Saving these values in case we want to use them for prediction
            )
            obj2goal_dist_list.append(prediction["relative_obj_to_goal"])

            pickup_bool = observations["pickedup_object"][i : (i + 1)]
            before_pickup = pickup_bool == 0  # not used because of our initialization
            after_pickup = pickup_bool == 1
            distances = arm2obj_dist
            distances[after_pickup] = obj2goal_dist[after_pickup]

            x = [distances, perception_embed[i : (i + 1)]]

            x_cat = torch.cat(x, dim=-1)
            x_out, rnn_hidden_states = self.state_encoder(
                x_cat, memory.tensor("rnn"), masks[i : (i + 1)]
            )
            memory = memory.set_tensor("rnn", rnn_hidden_states)

            x_out_list.append(x_out)

        x_out = torch.cat(x_out_list, dim=0)
        actor_out = self.actor(x_out)
        critic_out = self.critic(x_out)
        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out, extras={}
        )
        actor_critic_output.extras['relative_agent_arm_to_obj_prediction'] = torch.cat(arm2obj_dist_list, dim=0) #TODO is this the right dimension?
        actor_critic_output.extras['relative_agent_obj_to_goal_prediction'] = torch.cat(obj2goal_dist_list, dim=0)

        return (
            actor_critic_output,
            memory,
        )
