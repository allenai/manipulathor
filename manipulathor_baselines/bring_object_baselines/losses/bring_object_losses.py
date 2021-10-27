"""Defining imitation losses for actor critic type models."""
import platform
import random
import statistics
from typing import Dict, cast

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput

from manipulathor_utils.debugger_util import ForkedPdb


class PredictBoxBCELsss(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def loss(  # type: ignore
            self,
            step_count: int,
            batch: ObservationType,
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):
        """Computes the Prediction Box loss.

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

        ForkedPdb().set_trace()

        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        extra_model_outputs = actor_critic_output.extras

        gt_relative_agent_arm_to_obj = observations['relative_agent_arm_to_obj']
        gt_relative_obj_to_goal = observations['relative_obj_to_goal']

        pred_agent_arm_to_obj = extra_model_outputs['relative_agent_arm_to_obj_prediction']
        pred_obj_to_goal = extra_model_outputs['relative_agent_obj_to_goal_prediction']

        assert gt_relative_agent_arm_to_obj.shape == pred_agent_arm_to_obj.shape
        assert gt_relative_obj_to_goal.shape == pred_obj_to_goal.shape
        ForkedPdb().set_trace()
        loss_function = torch.nn.SmoothL1Loss() #LATER_TODO is this a good choice?
        arm_to_obj_loss = loss_function(gt_relative_agent_arm_to_obj, pred_agent_arm_to_obj)
        obj_to_goal_loss = loss_function(gt_relative_obj_to_goal, pred_obj_to_goal)
        total_loss = arm_to_obj_loss + obj_to_goal_loss



        return (
            total_loss,
            {"pred_box_bce": total_loss.item(),}
        )


class BinaryArmDistanceLoss(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()

        # #TODO this is literally a nightmare but oh well
        # self.average = {'closer':[], 'further':[]}

    def loss(  # type: ignore
            self,
            step_count: int,
            batch: ObservationType,
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):
        """Computes the Prediction Box loss.

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
        #TODO double check this

        is_object_visible = observations['is_goal_object_visible']
        binary_arm_distance = actor_critic_output.extras['binary_arm_distance']
        prev_arm_distance = observations['relative_arm_dist'][:-1]
        current_arm_distance = observations['relative_arm_dist'][1:]
        arm_distance = (current_arm_distance - prev_arm_distance)
        gt_binary_arm_distance = arm_distance < 0
        gt_binary_arm_distance = gt_binary_arm_distance.long()

        mask_over_actions = observations['previous_action_taken'].clone()
        mask_over_actions = mask_over_actions[1:]

        binary_arm_distance = binary_arm_distance[1:]

        object_is_visible = is_object_visible[1:]
        mask_over_actions[~object_is_visible] = 0
        action_exist = mask_over_actions.sum(dim=-1) != 0
        gt_binary_arm_distance = gt_binary_arm_distance[action_exist]
        masked_arm_dis = binary_arm_distance[mask_over_actions]

        # num_steps, workers, num_actions = mask_over_actions.shape
        # gt_binary_arm_distance = gt_binary_arm_distance.view(num_steps * workers) #
        # mask_over_actions = mask_over_actions.view(num_steps * workers, num_actions)
        # binary_arm_distance = binary_arm_distance.view(num_steps * workers, num_actions)
        # masked_arm_dis = binary_arm_distance[mask_over_actions]


        #TODO weights?
        if not torch.any(action_exist):
            total_loss = torch.tensor(0)
        else:
            total_loss = self.criterion(masked_arm_dis, gt_binary_arm_distance)


        # print('arm_distance < 0')
        # print(arm_distance < 0)
        # print('gt_binary_arm_distance')
        # print(gt_binary_arm_distance)
        #

        # TODO have you seen a real nightmare come true?
        # if random.random() < 0.01 or platform.system() == "Darwin": # TODO if this is too slow convert to tensor and set a limit on how many it can hold
        #     with torch.no_grad():
        #         if torch.any(action_exist):
        #             predicted_class = torch.argmax(masked_arm_dis, dim=-1)
        #             closer_distance = gt_binary_arm_distance == 1
        #             corrects = predicted_class == gt_binary_arm_distance
        #             self.average['closer'] += corrects[closer_distance].float().tolist()
        #             self.average['further'] += corrects[~ closer_distance].float().tolist()
        #             if random.random() < 0.5 or platform.system() == "Darwin":
        #                 print('closer', statistics.mean(self.average['closer']), len(self.average['closer']))
        #                 print('further', statistics.mean(self.average['further']), len(self.average['further']))

        return (
            total_loss,
            {"binary_arm_dist": total_loss.item(),}
        )

class FakeMaskDetectorLoss(AbstractActorCriticLoss):
    def __init__(self, noise, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = noise
        fake_mask_rates = self.noise * 2
        self.weights = torch.Tensor([1 / fake_mask_rates, 1 / (1 - fake_mask_rates)])
        self.criterion = None


    def loss(  # type: ignore
            self,
            step_count: int,
            batch: ObservationType,
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):



        observations = cast(Dict[str, torch.Tensor], batch["observations"])


        pickup_bool = observations["pickedup_object"]
        after_pickup = pickup_bool == 1

        is_real_mask_source_gt = observations['object_mask_source']['is_real_mask']
        is_real_mask_destination_gt = observations['object_mask_destination']['is_real_mask']
        is_real_mask_gt = is_real_mask_source_gt.clone()
        is_real_mask_gt[after_pickup] = is_real_mask_destination_gt[after_pickup]
        is_real_mask_gt = is_real_mask_gt.long()

        is_real_mask_pred = actor_critic_output.extras['is_real_mask']

        seq_len, b_size, num_cls = is_real_mask_pred.shape
        gt_seq_len, gt_bsize = is_real_mask_gt.shape
        assert seq_len == gt_seq_len and b_size == gt_bsize

        is_real_mask_pred = is_real_mask_pred.view(seq_len * b_size, num_cls)
        is_real_mask_gt = is_real_mask_gt.view(seq_len * b_size)

        if self.criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss(self.weights.to(is_real_mask_pred.device))

        total_loss = self.criterion(is_real_mask_pred, is_real_mask_gt)

        #TODO do we want to take care of cases where there is no mask and it's all zero? if yes should we change the weight?

        return (
            total_loss,
            {"fake_mask_detector_loss": total_loss.item(),}
        )

class MaskLoss(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def loss(  # type: ignore
            self,
            step_count: int,
            batch: ObservationType,
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):
        """Computes the Mask Loss

        """


        ForkedPdb().set_trace()
        observations = cast(Dict[str, torch.Tensor], batch["observations"])
        extra_model_outputs = actor_critic_output.extras

        predicted_mask = extra_model_outputs['predicted_mask']
        pickup_sensor = observations['pickedup_object']

        gt_masks_destination = observations['gt_mask_for_loss_destination']
        gt_masks_source = observations['gt_mask_for_loss_source']

        gt_masks = gt_masks_source
        gt_masks[pickup_sensor] = gt_masks_destination[pickup_sensor]

        ForkedPdb().set_trace()
        # all_masks = observations['all_masks_sensor'] TODO use this and object_category_source and object_category_destination for more failure analysis later

        intersection = (gt_masks + predicted_mask) == 2
        union = (gt_masks + predicted_mask) > 0
        interaction_sum = intersection.sum(dim=-1).sum(dim=-1).sum(dim=-1)
        union_sum = union.sum(dim=-1).sum(dim=-1).sum(dim=-1) + 1e-9
        mean_iou = (interaction_sum / union_sum).mean()
        total_loss = mean_iou
        #
        # gt_relative_agent_arm_to_obj = observations['relative_agent_arm_to_obj']
        # gt_relative_obj_to_goal = observations['relative_obj_to_goal']
        #
        # pred_agent_arm_to_obj = extra_model_outputs['relative_agent_arm_to_obj_prediction']
        # pred_obj_to_goal = extra_model_outputs['relative_agent_obj_to_goal_prediction']
        #
        # assert gt_relative_agent_arm_to_obj.shape == pred_agent_arm_to_obj.shape
        # assert gt_relative_obj_to_goal.shape == pred_obj_to_goal.shape
        # ForkedPdb().set_trace()
        # loss_function = torch.nn.SmoothL1Loss() #LATER_TODO is this a good choice?
        # arm_to_obj_loss = loss_function(gt_relative_agent_arm_to_obj, pred_agent_arm_to_obj)
        # obj_to_goal_loss = loss_function(gt_relative_obj_to_goal, pred_obj_to_goal)
        # total_loss = arm_to_obj_loss + obj_to_goal_loss



        return (
            total_loss,
            {"mask_loss": total_loss.item(),}
        )
