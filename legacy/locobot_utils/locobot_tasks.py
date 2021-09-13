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
    DONE,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb


class LocoBotBringObjectTask(BringObjectTask):
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

        self.manual = False
        if self.manual:
            action_str = 'something'
            actions = ('MoveArmHeightP', 'MoveArmHeightM', 'MoveArmXP', 'MoveArmXM', 'MoveArmYP', 'MoveArmYM', 'MoveArmZP', 'MoveArmZM', 'MoveAheadContinuous', 'RotateRightContinuous', 'RotateLeftContinuous')
            actions_short  = ('u', 'j', 's', 'a', '3', '4', 'w', 'z', 'm', 'r', 'l')
            action = 'm'
            self.env.controller.step('Pass')
            ForkedPdb().set_trace()
            action_str = actions[actions_short.index(action)]

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

        # if not self.object_picked_up:
        #     if object_id in self.env.controller.last_event.metadata['arm']['pickupableObjects']:
        #         event = self.env.step(dict(action="PickupObject"))
        #         #  we are doing an additional pass here, label is not right and if we fail we will do it twice
        #         object_inventory = self.env.controller.last_event.metadata["arm"][
        #             "heldObjects"
        #         ]
        #         if (
        #                 len(object_inventory) > 0
        #                 and object_id not in object_inventory
        #         ):
        #             event = self.env.step(dict(action="ReleaseObject"))
        #
        #     if self.env.is_object_at_low_level_hand(object_id):
        #         self.object_picked_up = True
        #         self.eplen_pickup = (
        #                 self._num_steps_taken + 1
        #         )  # plus one because this step has not been counted yet
        #
        # if self.object_picked_up:
        #
        #
        #     # self._took_end_action = True
        #     # self.last_action_success = True
        #     # self._success = True
        #
        #     source_state = self.env.get_object_by_id(object_id)
        #     goal_state = self.env.get_object_by_id(self.task_info['goal_object_id'])
        #     goal_achieved = self.object_picked_up and self.objects_close_enough(
        #         source_state, goal_state
        #     )
        #     if goal_achieved:
        #         self._took_end_action = True
        #         self.last_action_success = goal_achieved
        #         self._success = goal_achieved

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