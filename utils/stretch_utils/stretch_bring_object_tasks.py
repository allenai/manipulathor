"""Task Definions for the task of ArmPointNav"""
import datetime
from typing import Dict, Tuple, List, Any, Optional

import gym
import numpy as np
import torch
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task

from ithor_arm.bring_object_tasks import BringObjectTask, position_distance
from ithor_arm.ithor_arm_constants import (
    MOVE_ARM_CONSTANT,
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_X_P,
    MOVE_ARM_X_M,
    MOVE_ARM_Y_P,
    MOVE_ARM_Y_M,
    MOVE_ARM_Z_P,
    MOVE_ARM_Z_M,
    MOVE_AHEAD,
    ROTATE_RIGHT,
    ROTATE_LEFT,
    PICKUP,
    DONE, ARM_LENGTH, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, ROTATE_RIGHT_SMALL, ROTATE_LEFT_SMALL, MOVE_WRIST_P_SMALL, MOVE_WRIST_M_SMALL,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.jupyter_helper import get_reachable_positions


class ExploreWiseRewardTaskObjNav(BringObjectTask):
    _actions = (
        # MOVE_ARM_HEIGHT_P,
        # MOVE_ARM_HEIGHT_M,
        # MOVE_ARM_Z_P,
        # MOVE_ARM_Z_M,
        MOVE_AHEAD,
        MOVE_BACK,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        # PICKUP,
        # DONE,
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_locations = [[k['x'], k['y'], k['z']] for k in get_reachable_positions(self.env.controller)]
        self.all_reachable_positions = torch.Tensor(all_locations)
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1)) # do something about rotation here
        self.source_observed_reward = False
        self.goal_observed_reward = False




    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]

        self.manual = False
        if self.manual:
            actions = self._actions
            actions_short = ('m', 'b', 'r', 'l' )
            action = 'm'
            ForkedPdb().set_trace()
            action_str = actions[actions_short.index(action)]


        self._last_action_str = action_str
        action_dict = {"action": action_str}
        object_id = self.task_info["source_object_id"]
        if action_str == PICKUP:
            action_dict = {**action_dict, "object_id": object_id}
        self.env.step(action_dict)
        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        last_action_success = float(self.last_action_success)
        self.action_sequence_and_success.append((last_action_name, last_action_success))
        self.visualize(last_action_name)

        if True:
            source_state = self.env.get_object_by_id(object_id)
            object_to_reach = self.env.get_object_by_id(self.task_info['source_object_id'])
            goal_achieved = object_to_reach['visible'] == True

            if goal_achieved:
                self._took_end_action = True
                self.last_action_success = goal_achieved
                self._success = goal_achieved

        else:
            if not self.object_picked_up:
                if object_id in self.env.controller.last_event.metadata['arm']['pickupableObjects']:
                    event = self.env.step(dict(action="PickupObject"))
                    #  we are doing an additional pass here, label is not right and if we fail we will do it twice
                    object_inventory = self.env.controller.last_event.metadata["arm"][
                        "heldObjects"
                    ]
                    if (
                            len(object_inventory) > 0
                            and object_id not in object_inventory
                    ):
                        event = self.env.step(dict(action="ReleaseObject"))

                if self.env.is_object_at_low_level_hand(object_id):
                    self.object_picked_up = True
                    self.eplen_pickup = (
                            self._num_steps_taken + 1
                    )  # plus one because this step has not been counted yet

            if self.object_picked_up:

                source_state = self.env.get_object_by_id(object_id)
                goal_state = self.env.get_object_by_id(self.task_info['goal_object_id'])
                goal_achieved = self.object_picked_up and self.objects_close_enough(
                    source_state, goal_state
                )
                if goal_achieved:
                    self._took_end_action = True
                    self.last_action_success = goal_achieved
                    self._success = goal_achieved

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = self.reward_configs["step_penalty"]


        current_agent_location = self.env.get_agent_location()
        current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
        all_distances = self.all_reachable_positions - current_agent_location
        all_distances = (all_distances ** 2).sum(dim=-1)
        location_index = torch.argmin(all_distances)
        if self.has_visited[location_index] == 0:
            visited_new_place = True
        else:
            visited_new_place = False
        self.has_visited[location_index] = 1

        if visited_new_place and not self.source_observed_reward:
            reward += self.reward_configs["exploration_reward"]
        elif visited_new_place and self.object_picked_up and not self.goal_observed_reward:
            reward += self.reward_configs["exploration_reward"]
        source_is_visible = self.env.last_event.get_object(self.task_info['source_object_id'])['visible']
        goal_is_visible = self.env.last_event.get_object(self.task_info['goal_object_id'])['visible']

        if not self.source_observed_reward and source_is_visible:
            self.source_observed_reward = True
            reward += self.reward_configs['object_found']

        if not self.goal_observed_reward and goal_is_visible:
            self.goal_observed_reward = True
            reward += self.reward_configs['object_found']

        if not self.last_action_success or (
                self._last_action_str == PICKUP and not self.object_picked_up
        ):
            reward += self.reward_configs["failed_action_penalty"]

        if self._took_end_action:
            reward += (
                self.reward_configs["goal_success_reward"]
                if self._success
                else self.reward_configs["failed_stop_reward"]
            )

        # increase reward if object pickup and only do it once
        if not self.got_reward_for_pickup and self.object_picked_up:
            reward += self.reward_configs["pickup_success_reward"]
            self.got_reward_for_pickup = True
        #



        current_obj_to_arm_distance = self.arm_distance_from_obj()
        if self.last_arm_to_obj_distance is None or self.last_arm_to_obj_distance > ARM_LENGTH * 2: # is this good?
            delta_arm_to_obj_distance_reward = 0
        else:
            delta_arm_to_obj_distance_reward = (
                    self.last_arm_to_obj_distance - current_obj_to_arm_distance
            )
        self.last_arm_to_obj_distance = current_obj_to_arm_distance
        reward += delta_arm_to_obj_distance_reward

        current_obj_to_goal_distance = self.obj_distance_from_goal()
        if self.last_obj_to_goal_distance is None or self.last_obj_to_goal_distance > ARM_LENGTH * 2:
            delta_obj_to_goal_distance_reward = 0
        else:
            delta_obj_to_goal_distance_reward = (
                    self.last_obj_to_goal_distance - current_obj_to_goal_distance
            )
        self.last_obj_to_goal_distance = current_obj_to_goal_distance
        reward += delta_obj_to_goal_distance_reward

        # add collision cost, maybe distance to goal objective,...

        return float(reward)

class StretchExploreWiseRewardTask(BringObjectTask):
    _actions = (
        MOVE_ARM_HEIGHT_P,
        MOVE_ARM_HEIGHT_M,
        MOVE_ARM_Z_P,
        MOVE_ARM_Z_M,
        MOVE_WRIST_P,
        MOVE_WRIST_M,
        MOVE_AHEAD,
        MOVE_BACK,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        ROTATE_RIGHT_SMALL,
        ROTATE_LEFT_SMALL,
        MOVE_WRIST_P_SMALL,
        MOVE_WRIST_M_SMALL,
        # PICKUP,
        # DONE,
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_locations = [[k['x'], k['y'], k['z']] for k in get_reachable_positions(self.env.controller)]
        self.all_reachable_positions = torch.Tensor(all_locations)
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
        self.source_observed_reward = False
        self.goal_observed_reward = False

        self.last_body_to_obj_distance = None
        self.agent_body_dist_to_obj = []



    def metrics(self) -> Dict[str, Any]:
        result = super(StretchExploreWiseRewardTask, self).metrics()
        if self.is_done():
            result['percent_room_visited'] = self.has_visited.mean().item()

        return result
    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]

        self.manual = False
        if self.manual:
            # actions = ()
            # actions_short  = ('u', 'j', 's', 'a', '3', '4', 'w', 'z', 'm', 'r', 'l')
            ARM_ACTIONS_ORDERED = [MOVE_ARM_HEIGHT_P,MOVE_ARM_HEIGHT_M,MOVE_ARM_Z_P,MOVE_ARM_Z_M,MOVE_WRIST_P,MOVE_WRIST_M,MOVE_AHEAD,MOVE_BACK,ROTATE_RIGHT,ROTATE_LEFT,]
            ARM_SHORTENED_ACTIONS_ORDERED = ['hp','hm','zp','zm','wp','wm','m', 'b','r','l']
            action = 'm'
            self.env.controller.step('Pass')
            print(self.task_info['source_object_id'], self.task_info['goal_object_id'], 'pickup', self.object_picked_up)
            ForkedPdb().set_trace()
            action_str = ARM_ACTIONS_ORDERED[ARM_SHORTENED_ACTIONS_ORDERED.index(action)]


        self._last_action_str = action_str
        action_dict = {"action": action_str}
        object_id = self.task_info["source_object_id"]
        if action_str == PICKUP:
            action_dict = {**action_dict, "object_id": object_id}
        self.env.step(action_dict)
        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        last_action_success = float(self.last_action_success)
        self.action_sequence_and_success.append((last_action_name, last_action_success))
        self.agent_body_dist_to_obj.append(self.body_distance_from_obj())
        self.visualize(last_action_name)

        if not self.object_picked_up:# and False:
            if object_id in self.env.controller.last_event.metadata['arm']['pickupableObjects']:
                event = self.env.step(dict(action="PickupObject"))
                #  we are doing an additional pass here, label is not right and if we fail we will do it twice
                object_inventory = self.env.controller.last_event.metadata["arm"][
                    "heldObjects"
                ]
                if (
                        len(object_inventory) > 0
                        and object_id not in object_inventory
                ):
                    event = self.env.step(dict(action="ReleaseObject"))

            if self.env.is_object_at_low_level_hand(object_id):
                self.object_picked_up = True
                self.eplen_pickup = (
                        self._num_steps_taken + 1
                )  # plus one because this step has not been counted yet

        if self.object_picked_up:

            source_state = self.env.get_object_by_id(object_id)
            goal_state = self.env.get_object_by_id(self.task_info['goal_object_id'])
            goal_achieved = self.object_picked_up and self.objects_close_enough(
                source_state, goal_state
            )
            if goal_achieved:
                self._took_end_action = True
                self.last_action_success = goal_achieved
                self._success = goal_achieved

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def body_distance_from_obj(self):
        source_object_id = self.task_info["source_object_id"]
        object_info = self.env.get_object_by_id(source_object_id)
        agent_state = dict(position={k:v for (k,v) in self.env.get_agent_location().items() if k in ['x','y', 'z']})
        return position_distance(object_info, agent_state)

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = self.reward_configs["step_penalty"]


        current_agent_location = self.env.get_agent_location()
        current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
        all_distances = self.all_reachable_positions - current_agent_location
        all_distances = (all_distances ** 2).sum(dim=-1)
        location_index = torch.argmin(all_distances)
        if self.has_visited[location_index] == 0:
            visited_new_place = True
        else:
            visited_new_place = False
        self.has_visited[location_index] = 1

        if visited_new_place and not self.source_observed_reward:
            reward += self.reward_configs["exploration_reward"]
        elif visited_new_place and self.object_picked_up and not self.goal_observed_reward:
            reward += self.reward_configs["exploration_reward"]

        source_is_visible = self.env.last_event.get_object(self.task_info['source_object_id'])['visible']
        goal_is_visible = self.env.last_event.get_object(self.task_info['goal_object_id'])['visible']

        if not self.source_observed_reward and source_is_visible:
            self.source_observed_reward = True
            reward += self.reward_configs['object_found']

        if not self.goal_observed_reward and goal_is_visible:
            self.goal_observed_reward = True
            reward += self.reward_configs['object_found']

        if not self.last_action_success or (
                self._last_action_str == PICKUP and not self.object_picked_up
        ):
            reward += self.reward_configs["failed_action_penalty"]

        if self._took_end_action:
            reward += (
                self.reward_configs["goal_success_reward"]
                if self._success
                else self.reward_configs["failed_stop_reward"]
            )

        # increase reward if object pickup and only do it once
        if not self.got_reward_for_pickup and self.object_picked_up:
            reward += self.reward_configs["pickup_success_reward"]
            self.got_reward_for_pickup = True
        #

        current_obj_to_arm_distance = self.arm_distance_from_obj()
        if self.last_arm_to_obj_distance is None or self.last_arm_to_obj_distance > ARM_LENGTH * 2: # is this good?
            delta_arm_to_obj_distance_reward = 0
        else:
            delta_arm_to_obj_distance_reward = (
                    self.last_arm_to_obj_distance - current_obj_to_arm_distance
            )
        self.last_arm_to_obj_distance = current_obj_to_arm_distance
        reward += delta_arm_to_obj_distance_reward * self.reward_configs["arm_dist_multiplier"]

        current_obj_to_goal_distance = self.obj_distance_from_goal()
        if self.last_obj_to_goal_distance is None or self.last_obj_to_goal_distance > ARM_LENGTH * 2:
            delta_obj_to_goal_distance_reward = 0
        else:
            delta_obj_to_goal_distance_reward = (
                    self.last_obj_to_goal_distance - current_obj_to_goal_distance
            )
        self.last_obj_to_goal_distance = current_obj_to_goal_distance * self.reward_configs["arm_dist_multiplier"]
        reward += delta_obj_to_goal_distance_reward

        current_obj_to_body_distance = self.body_distance_from_obj()
        if self.last_body_to_obj_distance is None: # is this good?
            delta_body_to_obj_distance_reward = 0
        else:
            delta_body_to_obj_distance_reward = (
                    self.last_body_to_obj_distance - current_obj_to_body_distance
            )
        self.last_body_to_obj_distance = current_obj_to_body_distance
        reward += delta_body_to_obj_distance_reward * self.reward_configs["arm_dist_multiplier"]

        # add collision cost, maybe distance to goal objective,...

        return float(reward)
#
# class StretchObjectNavTask(StretchExploreWiseRewardTask):
#     _actions = (
#         MOVE_AHEAD,
#         MOVE_BACK,
#         ROTATE_RIGHT,
#         ROTATE_LEFT,
#         # ROTATE_RIGHT_SMALL,
#         # ROTATE_LEFT_SMALL,
#         # PICKUP,
#         # DONE,
#     )
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         all_locations = [[k['x'], k['y'], k['z']] for k in get_reachable_positions(self.env.controller)]
#         self.all_reachable_positions = torch.Tensor(all_locations)
#         self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
#         self.source_observed_reward = False
#         self.goal_observed_reward = False
#
#
#     def metrics(self) -> Dict[str, Any]:
#         result = super(StretchObjectNavTask, self).metrics()
#         if self.is_done():
#             result['percent_room_visited'] = self.has_visited.mean().item()
#
#         return result
#     def _step(self, action: int) -> RLStepResult:
#
#         action_str = self.class_action_names()[action]
#
#         self.manual = False
#         if self.manual:
#             # actions = ()
#             # actions_short  = ('u', 'j', 's', 'a', '3', '4', 'w', 'z', 'm', 'r', 'l')
#             ARM_ACTIONS_ORDERED = [MOVE_ARM_HEIGHT_P,MOVE_ARM_HEIGHT_M,MOVE_ARM_Z_P,MOVE_ARM_Z_M,MOVE_WRIST_P,MOVE_WRIST_M,MOVE_AHEAD,MOVE_BACK,ROTATE_RIGHT,ROTATE_LEFT,]
#             ARM_SHORTENED_ACTIONS_ORDERED = ['hp','hm','zp','zm','wp','wm','m', 'b','r','l']
#             action = 'm'
#             self.env.controller.step('Pass')
#             print(self.task_info['source_object_id'], self.task_info['goal_object_id'], 'pickup', self.object_picked_up)
#             ForkedPdb().set_trace()
#             action_str = ARM_ACTIONS_ORDERED[ARM_SHORTENED_ACTIONS_ORDERED.index(action)]
#
#
#         self._last_action_str = action_str
#         action_dict = {"action": action_str}
#         object_id = self.task_info["source_object_id"]
#         if action_str == PICKUP:
#             action_dict = {**action_dict, "object_id": object_id}
#         self.env.step(action_dict)
#         self.last_action_success = self.env.last_action_success
#
#         last_action_name = self._last_action_str
#         last_action_success = float(self.last_action_success)
#         self.action_sequence_and_success.append((last_action_name, last_action_success))
#         self.agent_body_dist_to_obj.append(self.body_distance_from_obj())
#         self.visualize(last_action_name)
#
#         object_visible = self.env.get_object_by_id(object_id)['visible']
#         if object_visible:
#             self._took_end_action = True
#             self.last_action_success = True
#             self._success = True
#
#
#         step_result = RLStepResult(
#             observation=self.get_observations(),
#             reward=self.judge(),
#             done=self.is_done(),
#             info={"last_action_success": self.last_action_success},
#         )
#         return step_result
#     def judge(self) -> float:
#         """Compute the reward after having taken a step."""
#         reward = self.reward_configs["step_penalty"]
#
#
#         current_agent_location = self.env.get_agent_location()
#         current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
#         all_distances = self.all_reachable_positions - current_agent_location
#         all_distances = (all_distances ** 2).sum(dim=-1)
#         location_index = torch.argmin(all_distances)
#         if self.has_visited[location_index] == 0:
#             visited_new_place = True
#         else:
#             visited_new_place = False
#         self.has_visited[location_index] = 1
#
#         if visited_new_place and not self.source_observed_reward:
#             reward += self.reward_configs["exploration_reward"]
#         elif visited_new_place and self.object_picked_up and not self.goal_observed_reward:
#             reward += self.reward_configs["exploration_reward"]
#
#         source_is_visible = self.env.last_event.get_object(self.task_info['source_object_id'])['visible']
#         goal_is_visible = self.env.last_event.get_object(self.task_info['goal_object_id'])['visible']
#
#         if not self.source_observed_reward and source_is_visible:
#             self.source_observed_reward = True
#             reward += self.reward_configs['object_found']
#
#         if not self.goal_observed_reward and goal_is_visible:
#             self.goal_observed_reward = True
#             reward += self.reward_configs['object_found']
#
#         if not self.last_action_success or (
#                 self._last_action_str == PICKUP and not self.object_picked_up
#         ):
#             reward += self.reward_configs["failed_action_penalty"]
#
#         if self._took_end_action:
#             reward += (
#                 self.reward_configs["goal_success_reward"]
#                 if self._success
#                 else self.reward_configs["failed_stop_reward"]
#             )
#
#         # increase reward if object pickup and only do it once
#         if not self.got_reward_for_pickup and self.object_picked_up:
#             reward += self.reward_configs["pickup_success_reward"]
#             self.got_reward_for_pickup = True
#         #
#
#         TODO this needs to be changed. We might need two separate ones for this
#         # current_obj_to_arm_distance = self.arm_distance_from_obj()
#         # if self.last_arm_to_obj_distance is None or self.last_arm_to_obj_distance > ARM_LENGTH * 2: # is this good?
#         #     delta_arm_to_obj_distance_reward = 0
#         # else:
#         #     delta_arm_to_obj_distance_reward = (
#         #             self.last_arm_to_obj_distance - current_obj_to_arm_distance
#         #     )
#         # self.last_arm_to_obj_distance = current_obj_to_arm_distance
#         # reward += delta_arm_to_obj_distance_reward * self.reward_configs["arm_dist_multiplier"]
#         # current_obj_to_goal_distance = self.obj_distance_from_goal()
#         # if self.last_obj_to_goal_distance is None or self.last_obj_to_goal_distance > ARM_LENGTH * 2:
#         #     delta_obj_to_goal_distance_reward = 0
#         # else:
#         #     delta_obj_to_goal_distance_reward = (
#         #             self.last_obj_to_goal_distance - current_obj_to_goal_distance
#         #     )
#         # self.last_obj_to_goal_distance = current_obj_to_goal_distance * self.reward_configs["arm_dist_multiplier"]
#         # reward += delta_obj_to_goal_distance_reward
#
#
#         current_obj_to_body_distance = self.body_distance_from_obj()
#         if self.last_body_to_obj_distance is None: # is this good?
#             delta_body_to_obj_distance_reward = 0
#         else:
#             delta_body_to_obj_distance_reward = (
#                     self.last_body_to_obj_distance - current_obj_to_body_distance
#             )
#         self.last_body_to_obj_distance = current_obj_to_body_distance
#         reward += delta_body_to_obj_distance_reward * self.reward_configs["arm_dist_multiplier"]
#
#
#         # add collision cost, maybe distance to goal objective,...
#
#         return float(reward)