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
from ithor_arm_environment import ManipulaTHOREnvironment

from ithor_arm.ithor_arm_constants import (
    ADITIONAL_ARM_ARGS,
    ARM_MIN_HEIGHT,
    ARM_MAX_HEIGHT,
    MOVE_ARM_HEIGHT_CONSTANT,
    MOVE_ARM_CONSTANT,
    MANIPULATHOR_COMMIT_ID,
    reset_environment_and_additional_commands,
    MOVE_THR, PICKUP, DONE, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, SET_OF_ALL_AGENT_ACTIONS,
)
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.hacky_objects_that_move import OBJECTS_MOVE_THR
from allenact.utils.misc_utils import prepare_locals_for_super


class ManipulaTHOREnvironmentNoisy(ManipulaTHOREnvironment):
    """Wrapper for the manipulathor controller providing arm functionality
    and bookkeeping.

    See [here](https://ai2thor.allenai.org/documentation/installation) for comprehensive
     documentation on AI2-THOR.

    # Attributes

    controller : The ai2thor controller.
    """

    def __init__(
            self,
            x_display: Optional[str] = None,
            ahead_noise_meta_dist_params: Dict[str, float] = {'bias_dist': [0,0], 'variance_dist': [0,0]},
            lateral_noise_meta_dist_params: Dict[str, float] = {'bias_dist': [0,0], 'variance_dist': [0,0]},
            turning_noise_meta_dist_params: Dict[str, float] = {'bias_dist': [0,0], 'variance_dist': [0,0]},
            **kwargs: Any
    ) -> None:
        """Initializer.

        # Parameters

        *_noise_meta_dist_params : [mean, variance] defines the normal distribution over which the actual noise parameters
        for a motion noise distribution will be drawn. Distributions for noise in motion will be re-rolled every scene 
        reset with new bias and variance values drawn from these meta-distributions.
        
        """

        self.ahead_noise_meta_dist_params = ahead_noise_meta_dist_params
        self.lateral_noise_meta_dist_params = lateral_noise_meta_dist_params
        self.turning_noise_meta_dist_params = turning_noise_meta_dist_params
        self.ahead_noise_params = [0,0]
        self.lateral_noise_params = [0,0]
        self.turning_noise_params = [0,0]

        super().__init__(**prepare_locals_for_super(locals()))

    def generate_motion_noise_params(self,meta_dist):
        bias = np.random.normal(*meta_dist['bias_dist'])
        variance = np.random.normal(*meta_dist['variance_dist'])
        return [bias,variance]

    
    def reset_agent_motion_noise_models(self):
        self.ahead_noise_params = self.generate_motion_noise_params(self,self.ahead_noise_meta_dist_params)
        self.lateral_noise_params = self.generate_motion_noise_params(self,self.lateral_noise_meta_dist_params)
        self.turning_noise_params = self.generate_motion_noise_params(self,self.turning_noise_meta_dist_params)


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
            self.controller = ai2thor.controller.Controller(
                **self.env_args
            )
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

        self.reset_agent_motion_noise_models()


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

        elif action in [MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT]:

            copy_aditions = copy.deepcopy(ADITIONAL_ARM_ARGS)

            action_dict = {**action_dict, **copy_aditions}
            if action in [MOVE_AHEAD]:
                action_dict["action"] = "MoveAgent"
                action_dict["ahead"] = 0.2 + np.random.Normal(*self.ahead_noise_params)

            elif action in [ROTATE_RIGHT]:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = 45

            elif action in [ROTATE_LEFT]:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = -45

        elif "MoveArm" in action:
            copy_aditions = copy.deepcopy(ADITIONAL_ARM_ARGS)
            action_dict = {**action_dict, **copy_aditions}
            base_position = self.get_current_arm_state()
            if "MoveArmHeight" in action:
                action_dict["action"] = "MoveArmBase"

                if action == MOVE_ARM_HEIGHT_P:
                    base_position["h"] += MOVE_ARM_HEIGHT_CONSTANT
                if action == MOVE_ARM_HEIGHT_M:
                    base_position[
                        "h"
                    ] -= MOVE_ARM_HEIGHT_CONSTANT  # height is pretty big!
                action_dict["y"] = base_position["h"]
            else:
                action_dict["action"] = "MoveArm"
                if action == MOVE_ARM_X_P:
                    base_position["x"] += MOVE_ARM_CONSTANT
                elif action == MOVE_ARM_X_M:
                    base_position["x"] -= MOVE_ARM_CONSTANT
                elif action == MOVE_ARM_Y_P:
                    base_position["y"] += MOVE_ARM_CONSTANT
                elif action == MOVE_ARM_Y_M:
                    base_position["y"] -= MOVE_ARM_CONSTANT
                elif action == MOVE_ARM_Z_P:
                    base_position["z"] += MOVE_ARM_CONSTANT
                elif action == MOVE_ARM_Z_M:
                    base_position["z"] -= MOVE_ARM_CONSTANT
                action_dict["position"] = {
                    k: v for (k, v) in base_position.items() if k in ["x", "y", "z"]
                }



        sr = self.controller.step(action_dict)
        self.list_of_actions_so_far.append(action_dict)

        if action in SET_OF_ALL_AGENT_ACTIONS:
            self.update_memory()

        if self._verbose:
            print(self.controller.last_event)

        if self.restrict_to_initially_reachable_points:
            self._snap_agent_to_initially_reachable()

        if skip_render:
            assert last_frame is not None
            self.last_event.frame = last_frame

        return sr

