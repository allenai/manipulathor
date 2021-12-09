"""A wrapper for engaging with the ManipulaTHOR environment."""

import copy
import typing
import warnings
from typing import Dict, Union, Optional

import ai2thor.server
import numpy as np
from ai2thor.controller import Controller

from ithor_arm.ithor_arm_constants import (
    ADITIONAL_ARM_ARGS,
    PICKUP, DONE, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, MOVE_BACK, MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, MOVE_WRIST_P, MOVE_WRIST_M,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.jupyter_helper import ARM_MOVE_CONSTANT
from scripts.stretch_jupyter_helper import get_current_arm_state, WRIST_ROTATION, reset_environment_and_additional_commands
from utils.stretch_utils.stretch_constants import STRETCH_MANIPULATHOR_COMMIT_ID


class StretchManipulaTHOREnvironment(ManipulaTHOREnvironment):
    """Wrapper for the manipulathor controller providing arm functionality
    and bookkeeping.

    See [here](https://ai2thor.allenai.org/documentation/installation) for comprehensive
     documentation on AI2-THOR.

    # Attributes

    controller : The ai2thor controller.
    """
    def reset(
            self,
            scene_name: Optional[str],
            move_mag: float = 0.25,
            **kwargs,
    ):
        self._move_mag = move_mag
        self._grid_size = self._move_mag
        # self.memory_frames = []

        if scene_name is None:
            scene_name = self.controller.last_event.metadata["sceneName"]
        # self.reset_init_params()#**kwargs) removing this fixes one of the crashing problem

        # to solve the crash issue
        # why do we still have this crashing problem?
        try:
            reset_environment_and_additional_commands(self.controller, scene_name)
        except Exception as e:
            print("RESETTING THE SCENE,", scene_name, 'because of', str(e))
            self.controller = self.create_controller()
            reset_environment_and_additional_commands(self.controller, scene_name)

        if self.object_open_speed != 1.0:
            self.controller.step(
                {"action": "ChangeOpenSpeed", "x": self.object_open_speed}
            )

        self._initially_reachable_points = None
        self._initially_reachable_points_set = None
        self.controller.step({"action": "GetReachablePositions"})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            warnings.warn(
                "Error when getting reachable points: {}".format(
                    self.controller.last_event.metadata["errorMessage"]
                )
            )
        self._initially_reachable_points = self.last_action_return

        self.list_of_actions_so_far = []


    def check_controller_version(self):
        if STRETCH_MANIPULATHOR_COMMIT_ID is not None:
            assert (
                STRETCH_MANIPULATHOR_COMMIT_ID in self.controller._build.url
            ), "Build number is not right, {} vs {}, use  pip3 install -e git+https://github.com/allenai/ai2thor.git@{}#egg=ai2thor".format(
                self.controller._build.url,
                STRETCH_MANIPULATHOR_COMMIT_ID,
                STRETCH_MANIPULATHOR_COMMIT_ID,
            )

    def create_controller(self):
        controller = Controller(**self.env_args, commit_id=STRETCH_MANIPULATHOR_COMMIT_ID)

        return controller

    @property
    def arm_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        return self.controller.last_event.third_party_camera_frames[0]

    def get_current_arm_state(self):
        arm = self.controller.last_event.metadata['arm']['joints'] #TODO is this the right one? how about wrist movements
        z = arm[-1]['rootRelativePosition']['z']
        x = 0 #arm[-1]['rootRelativePosition']['x']
        y = arm[0]['rootRelativePosition']['y'] - 0.16297650337219238 #TODO?
        return dict(x=0,y=y, z=z)


    def get_absolute_hand_state(self):
        # event = self.controller.last_event
        # joints = event.metadata["arm"]["joints"]
        # arm = copy.deepcopy(joints[-1])
        # xyz_dict = arm["position"]
        xyz_dict = get_current_arm_state(self.controller)
        return dict(position=xyz_dict, rotation={"x": 0, "y": 0, "z": 0})

    def get_pickupable_objects(self):

        event = self.controller.last_event
        object_list = event.metadata["arm"]["pickupableObjects"]

        return object_list

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""

        action = typing.cast(str, action_dict["action"])

        skip_render = "renderImage" in action_dict and not action_dict["renderImage"]
        last_frame: Optional[np.ndarray] = None
        if skip_render:
            last_frame = self.current_frame

        if self.simplify_physics:
            action_dict["simplifyOPhysics"] = True
        if action in [PICKUP, DONE]:
            if action == PICKUP:
                object_id = action_dict["object_id"]
                if not self.is_object_at_low_level_hand(object_id):
                    pickupable_objects = self.get_pickupable_objects()
                    #
                    if object_id in pickupable_objects:
                        # This version of the task is actually harder # consider making it easier, are we penalizing failed pickup? yes
                        event = self.step(dict(action="PickupObject"))
                        #  we are doing an additional pass here, label is not right and if we fail we will do it twice
                        object_inventory = self.controller.last_event.metadata["arm"][
                            "heldObjects"
                        ]
                        if (
                            len(object_inventory) > 0
                            and object_id not in object_inventory
                        ):
                            print('Picked up the wrong object')
                            event = self.step(dict(action="ReleaseObject"))
            action_dict = {
                'action': 'Pass'
            } # we have to change the last action success if the pik up fails, we do that in the task now

        elif action in [MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT]:
            copy_aditions = copy.deepcopy(ADITIONAL_ARM_ARGS)
            action_dict = {**action_dict, **copy_aditions}
            if action == MOVE_AHEAD:
                action_dict["action"] = "MoveAgent"
                action_dict["ahead"] = 0.2 #TODO replace with constant equal to real world stuff
            elif action == MOVE_BACK:
                action_dict["action"] = "MoveAgent"
                action_dict["ahead"] = -0.2 #TODO we might have to mess with this and arm constant as well
            elif action == ROTATE_RIGHT:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = 45 #TODO we might have to make this smaller.

            elif action == ROTATE_LEFT:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = -45
        elif action in [MOVE_ARM_HEIGHT_P,MOVE_ARM_HEIGHT_M,MOVE_ARM_Z_P,MOVE_ARM_Z_M,]:
            base_position = get_current_arm_state(self.controller)
            change_value = ARM_MOVE_CONSTANT
            if action == MOVE_ARM_HEIGHT_P:
                base_position['y'] += change_value
            elif action == MOVE_ARM_HEIGHT_M:
                base_position['y'] -= change_value
            elif action == MOVE_ARM_Z_P:
                base_position['z'] += change_value
            elif action == MOVE_ARM_Z_M:
                base_position['z'] -= change_value
            action_dict = dict(action='MoveArm', position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**ADITIONAL_ARM_ARGS)
        elif action in [MOVE_WRIST_P,MOVE_WRIST_M,]:
            if action == MOVE_WRIST_P:
                action_dict = dict(action='RotateWristRelative', yaw=-WRIST_ROTATION)
            elif action == MOVE_WRIST_M:
                action_dict = dict(action='RotateWristRelative', yaw=WRIST_ROTATION)

        sr = self.controller.step(action_dict)
        self.list_of_actions_so_far.append(action_dict)

        if self._verbose:
            print(self.controller.last_event)

        if self.restrict_to_initially_reachable_points:
            self._snap_agent_to_initially_reachable()

        if skip_render:
            assert last_frame is not None
            self.last_event.frame = last_frame

        return sr
