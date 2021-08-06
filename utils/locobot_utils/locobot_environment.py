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
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_utils.debugger_util import ForkedPdb
import ai2thor.robot_controller


class LocoBotEnvironment(ManipulaTHOREnvironment):
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
        controller = ai2thor.robot_controller.Controller(host="192.168.0.80", port=9000)
        controller.step('Initialize')
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
        #TODO do we need this?
        # self.reset(scene_name=scene_name, move_mag=move_mag, **kwargs)

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

        if action in ["PickUpMidLevel", "DoneMidLevel"] or 'MoveArm' in action:
            sr = False
            # self.controller.step('Initialize')
            self.last_image_changed = False
            return sr

        else:
            if action == 'MoveAheadContinuous':
                locobot_action = 'MoveAhead'
            elif action == 'RotateRightContinuous':
                locobot_action = 'RotateRight'
            elif action == 'RotateLeftContinuous':
                locobot_action = 'RotateLeft'
            else:
                raise Exception('Action not supported')
            self.last_image_changed = True

        sr = self.controller.step(locobot_action)
        self.list_of_actions_so_far.append(action)

        if self._verbose:
            print(self.controller.last_event)

        return sr
