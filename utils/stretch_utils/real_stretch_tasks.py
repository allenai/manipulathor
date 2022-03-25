"""Task Definions for the task of ArmPointNav"""

from typing import Dict, Tuple, List, Any, Optional

import gym
import numpy as np
import torch
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task

from ithor_arm.bring_object_tasks import BringObjectTask
from utils.stretch_utils.stretch_bring_object_tasks import StretchExploreWiseRewardTask
from utils.stretch_utils.stretch_constants import (
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
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb


class RealStretchExploreWiseRewardTask(StretchExploreWiseRewardTask):
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
    def set_reachable_positions(self):
        self.all_reachable_positions = torch.zeros((100,3))
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
    def metrics(self) -> Dict[str, Any]:
        result = {}
        if self.is_done():
            result = {**result, **self.calc_action_stat_metrics()}

            result["success"] = self._success

            self.finish_visualizer_metrics(result)
            self.finish_visualizer(self._success)
            self.action_sequence_and_success = []

        return result

    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]
        print('Model Said', action_str)

        self.manual_action = True
        # self.env.kinect_depth
        if self.manual_action:
            ARM_ACTIONS_ORDERED = [MOVE_ARM_HEIGHT_P,MOVE_ARM_HEIGHT_M,MOVE_ARM_Z_P,MOVE_ARM_Z_M,MOVE_WRIST_P,MOVE_WRIST_M,MOVE_AHEAD,MOVE_BACK,ROTATE_RIGHT,ROTATE_LEFT,]
            ARM_SHORTENED_ACTIONS_ORDERED = ['hp','hm','zp','zm','wp','wm','m', 'b','r','l']
            action = ''
            while(True):
                ForkedPdb().set_trace()
                try:
                    action_str = ARM_ACTIONS_ORDERED[ARM_SHORTENED_ACTIONS_ORDERED.index(action)]
                    break
                except Exception:
                    print("wrong action")
                    continue




        print('Action Called', action_str)

        self._last_action_str = action_str
        action_dict = {"action": action_str}
        # object_id = self.task_info["source_object_id"]
        # if action_str == PICKUP:
        #     action_dict = {**action_dict, "object_id": object_id}
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


    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = 0
        return reward
