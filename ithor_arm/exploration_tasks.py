"""Task Definions for the task of ArmPointNav"""

from typing import Dict, Tuple, List, Any, Optional

import gym
import numpy as np
import torch
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task

from ithor_arm.bring_object_tasks import AbstractBringObjectTask, BringObjectTask
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
    MOVE_LEFT,
    MOVE_RIGHT,
    ROTATE_RIGHT,
    ROTATE_LEFT,
    LOOK_UP,
    LOOK_DOWN,
    PICKUP,
    DONE, ARM_LENGTH, MOVE_BACK,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.jupyter_helper import get_reachable_positions


class ExploreTask(BringObjectTask):
    _actions = (
        MOVE_BACK,
        MOVE_AHEAD,
        MOVE_LEFT,
        MOVE_RIGHT,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        LOOK_UP,
        LOOK_DOWN
        #NOTE @samir add look up and look down here and also add it to the environment
    )
    def obj_distance_from_goal(self):
        return 0
    def arm_distance_from_obj(self):
        return 0
    def get_original_object_distance(self):
        return 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_locations = [[k['x'], k['y'], k['z']] for k in get_reachable_positions(self.env.controller)]
        self.all_reachable_positions = torch.Tensor(all_locations) #TODO @samir you can change this to cover more than just the lcoations and add observed objects as well
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
        self.seen_objects = set()
    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = self.reward_configs["step_penalty"]

        # TODO @samir add reward for objects
        # current_agent_location = self.env.get_agent_location()
        # current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
        # all_distances = self.all_reachable_positions - current_agent_location
        # all_distances = (all_distances ** 2).sum(dim=-1)
        # location_index = torch.argmin(all_distances)
        # if self.has_visited[location_index] == 0:
        #     reward += self.reward_configs["exploration_reward"]
        # self.has_visited[location_index] = 1

        # TODO: mess with this
        for o in self.env.visible_objects():
            if o not in self.seen_objects:
                reward += self.reward_configs["object_reward"]
            self.seen_objects.add(o)


        if not self.last_action_success:
            reward += self.reward_configs["failed_action_penalty"]

        if self._took_end_action:
            reward += (
                self.reward_configs["goal_success_reward"]
                if self._success
                else self.reward_configs["failed_stop_reward"]
            )

        # add collision cost, maybe distance to goal objective,...
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        result = super(AbstractBringObjectTask, self).metrics()
        if self.is_done():
            result['percent_room_visited'] = self.has_visited.mean()
            result["success"] = self._success
            # TODO @samir add metric for obect, logged in tb automatically
            self.finish_visualizer_metrics(result)
            self.finish_visualizer(self._success)
            self.action_sequence_and_success = []
        return result
    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]

        self._last_action_str = action_str
        action_dict = {"action": action_str}

        self.env.step(action_dict)
        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        last_action_success = float(self.last_action_success)
        self.action_sequence_and_success.append((last_action_name, last_action_success))
        self.visualize(last_action_name)


        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result


