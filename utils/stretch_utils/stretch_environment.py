"""A wrapper for engaging with the ManipulaTHOR environment."""

import copy
import math
import typing
import warnings
from typing import Tuple, Dict, List, Set, Union, Any, Optional

import ai2thor.server
import numpy as np
from ai2thor.controller import Controller
from allenact_plugins.ithor_plugin.ithor_constants import VISIBILITY_DISTANCE, FOV
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from ithor_arm.ithor_arm_constants import (
    ADITIONAL_ARM_ARGS,
    ARM_MIN_HEIGHT,
    ARM_MAX_HEIGHT,
    MOVE_ARM_HEIGHT_CONSTANT,
    MOVE_ARM_CONSTANT,
    MANIPULATHOR_COMMIT_ID,
    reset_environment_and_additional_commands,
    MOVE_THR,
    MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, PICKUP, DONE, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, GRASP_O, GRASP_C
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_utils.debugger_util import ForkedPdb
import ai2thor.robot_controller


class StretchEnvironment(ManipulaTHOREnvironment):
    """Wrapper for the manipulathor controller providing arm functionality
    and bookkeeping.

    See [here](https://ai2thor.allenai.org/documentation/installation) for comprehensive
     documentation on AI2-THOR.

    # Attributes

    controller : The ai2thor controller.
    """
    def check_controller_version(self):
        return True


    def create_controller(self):
        controller = ai2thor.robot_controller.Controller(host="stretch1.corp.ai2", port=9000, width=1280, height=720) #TODO frame width and height?
        # controller.step('Initialize') #TODO should i put this back?>
        return controller

    def start(
            self,
            scene_name: Optional[str],
            move_mag: float = 0.25,
            **kwargs,
    ) -> None:

        """Starts the ai2thor controller if it was previously stopped.

        After starting, `reset` will be called with the scene name and move magnitude.

        # Parameters

        scene_name : The scene to load.
        move_mag : The amount of distance the agent moves in a single `MoveAhead` step.
        kwargs : additional kwargs, passed to reset.
        """
        if self._started:
            raise RuntimeError(
                "Trying to start the environment but it is already started."
            )

        self.controller = self.create_controller()

        # if (
        #         self._start_player_screen_height,
        #         self._start_player_screen_width,
        # ) != self.current_frame.shape[:2]:
        #     self.controller.step(
        #         {
        #             "action": "ChangeResolution",
        #             "x": self._start_player_screen_width,
        #             "y": self._start_player_screen_height,
        #         }
        #     )

        self._started = True
        self.last_image_changed = True
        self.reset(scene_name=scene_name, move_mag=move_mag, **kwargs)

    def reset(
            self,
            scene_name: Optional[str],
            move_mag: float = 0.25,
            **kwargs,
    ):
        self._move_mag = move_mag
        self._grid_size = self._move_mag

        #TODO is this good?
        self.controller = self.create_controller()
        self.controller.step('Initialize')

        self.list_of_actions_so_far = []
        self.last_image_changed = True


    def step(
            self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        action = typing.cast(str, action_dict["action"])

        skip_render = "renderImage" in action_dict and not action_dict["renderImage"]
        last_frame: Optional[np.ndarray] = None
        if skip_render:
            last_frame = self.current_frame

        self.last_image_changed = True


        if action == MOVE_ARM_HEIGHT_P:
            stretch_action = dict(action='MoveArmBaseP',)
        elif action == MOVE_ARM_HEIGHT_M:
            stretch_action = dict(action='MoveArmBaseN',)
        elif action == MOVE_ARM_Z_P:
            stretch_action = dict(action='MoveArmTargetP',)
        elif action == MOVE_ARM_Z_M:
            stretch_action = dict(action='MoveArmTargetN',)
        elif action == MOVE_WRIST_P:
            stretch_action = dict(action='MoveWristP',)
        elif action == MOVE_WRIST_M:
            stretch_action = dict(action='MoveWristN',)
        elif action == GRASP_O:
            stretch_action = dict(action='GraspOpen',)
        elif action == GRASP_C:
            stretch_action = dict(action='GrapClose',)
        # elif action == MOVE_AHEAD:
        #     stretch_action = dict(action='MoveAhead',)
        # elif action == MOVE_BACK:
        #     stretch_action = dict(action='MoveBack',)
        # elif action == ROTATE_RIGHT:
        #     stretch_action = dict(action='RotateRight',)
        # elif action == ROTATE_LEFT:
        #     stretch_action = dict(action='RotateLeft',)
        else:
            print('Action Not Supported')
            self.last_image_changed = False
            stretch_action = dict(action='Pass',)
            # raise Exception('Action not supported')


        sr = self.controller.step(**stretch_action)
        self.list_of_actions_so_far.append(action)

        if self._verbose:
            print(self.controller.last_event)

        return sr
