"""Task Definions for the task of ArmPointNav"""

from typing import Dict, Tuple, List, Any, Optional

import gym
import numpy as np
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task

from ithor_arm.bring_object_tasks import BringObjectTask
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
    DONE, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, GRASP_O, GRASP_C,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb


class StretchRealBringObjectTask(BringObjectTask):
    _actions = (
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
        # PICKUP,
        # DONE,
    )
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

        self.manual_action = False
        if self.manual_action:
            list_of_actions = [MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, PICKUP, DONE, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, GRASP_O, GRASP_C]
            corespond = ['hp', 'hm', 'xp', 'xm', 'yp', 'ym', 'zp', 'zm', 'm', 'r', 'l', 'p', 'd', 'b', 'wp', 'wm', 'go', 'gc']
            action = ''
            ForkedPdb().set_trace()
            if action != '':
                action_str = list_of_actions[corespond.index(action)]

        # list_of_actions = [MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, PICKUP, DONE, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, GRASP_O, GRASP_C]
        # translate = [MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_Z_P, MOVE_ARM_Z_P, MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, MOVE_ARM_Z_P, ROTATE_RIGHT, ROTATE_LEFT, PICKUP, DONE, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, GRASP_O, GRASP_C]
        # translated_action = translate[list_of_actions.index(action_str)]
        # action_str = translated_action TODO

        # import matplotlib.pyplot as plt; plt.imsave('something.png', self.env.last_event.frame)

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

class StretchRealObjectNavTask(StretchRealBringObjectTask):
    _actions = (
        MOVE_AHEAD,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        MOVE_BACK
    )