"""Defining imitation losses for actor critic type models."""

from typing import Dict, cast

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput

from manipulathor_utils.debugger_util import ForkedPdb


class PredictDistanceLoss(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def loss(  # type: ignore
            self,
            step_count: int,
            batch: ObservationType,
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):
        """Computes the Prediction Distance loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.recurrent_generator`.
            Here `batch["observations"]` must contain `"expert_action"` observations
            or `"expert_policy"` observations. See `ExpertActionSensor` (or `ExpertPolicySensor`) for an example of
            a sensor producing such observations.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args. Ignored.
        kwargs : Extra kwargs. Ignored.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        """

        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        extra_model_outputs = actor_critic_output.extras

        gt_relative_agent_arm_to_obj = observations['relative_agent_arm_to_obj']
        gt_relative_obj_to_goal = observations['relative_obj_to_goal']

        pred_agent_arm_to_obj = extra_model_outputs['relative_agent_arm_to_obj_prediction']
        pred_obj_to_goal = extra_model_outputs['relative_agent_obj_to_goal_prediction']

        assert gt_relative_agent_arm_to_obj.shape == pred_agent_arm_to_obj.shape
        assert gt_relative_obj_to_goal.shape == pred_obj_to_goal.shape

        loss_function = torch.nn.SmoothL1Loss() TODO is this a good choice?
        arm_to_obj_loss = loss_function(gt_relative_agent_arm_to_obj, pred_agent_arm_to_obj)
        obj_to_goal_loss = loss_function(gt_relative_obj_to_goal, pred_obj_to_goal)
        total_loss = arm_to_obj_loss + obj_to_goal_loss



        return (
            total_loss,
            {"distance_pred_loss": total_loss.item(),}
        )
