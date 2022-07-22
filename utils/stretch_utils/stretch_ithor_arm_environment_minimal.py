import typing
from typing import Dict, Union, Optional
import numpy as np

import ai2thor.server
from ai2thor.controller import Controller
from allenact.utils.system import get_logger

from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from utils.stretch_utils.stretch_constants import (
    DONE, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, MOVE_BACK, 
)
from scripts.stretch_jupyter_helper import AGENT_MOVEMENT_CONSTANT, AGENT_ROTATION_DEG, remove_nan_inf_for_frames
from utils.stretch_utils.stretch_sim2real_utils import kinect_reshape, intel_reshape


class MinimalStretchManipulaTHOREnvironment(ManipulaTHOREnvironment):

    @property
    def kinect_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        frame = self.controller.last_event.third_party_camera_frames[0].copy()
        frame = remove_nan_inf_for_frames(frame, 'kinect_frame')
        return kinect_reshape(frame)
    @property
    def kinect_depth(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        if self.controller.last_event.third_party_depth_frames[0]:
            return None
        depth_frame = self.controller.last_event.third_party_depth_frames[0].copy()
        depth_frame = remove_nan_inf_for_frames(depth_frame, 'depth_kinect')

        # #TODO remove
        if np.sum(depth_frame != self.controller.last_event.third_party_depth_frames[0].copy()) > 10:
            raise Exception('Depth is nan again even after removing nan?')

        return kinect_reshape(depth_frame)

    @property
    def intel_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        frame = self.controller.last_event.frame.copy()
        frame = remove_nan_inf_for_frames(frame, 'intel_frame')
        return intel_reshape(frame)
    @property
    def intel_depth(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        if self.controller.last_event.depth_frame is None:
            return None
        depth_frame = self.controller.last_event.depth_frame.copy()
        depth_frame = remove_nan_inf_for_frames(depth_frame, 'depth_intel')
        return intel_reshape(depth_frame)

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
        if action in [DONE]:
            action_dict = {
                'action': 'Pass'
            } # we have to change the last action success if the pik up fails, we do that in the task now

        elif action in [MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT]:
            if action == MOVE_AHEAD:
                action_dict["action"] = "MoveAgent"
                action_dict["ahead"] = AGENT_MOVEMENT_CONSTANT
            elif action == MOVE_BACK:
                action_dict["action"] = "MoveAgent"
                action_dict["ahead"] = -AGENT_MOVEMENT_CONSTANT
            elif action == ROTATE_RIGHT:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = AGENT_ROTATION_DEG

            elif action == ROTATE_LEFT:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = -AGENT_ROTATION_DEG
        

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

    def create_controller(self):
        assert 'commit_id' in self.env_args, 'No commit id is specified'
        controller = Controller(**self.env_args)
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

            if (
                    self._start_player_screen_height,
                    self._start_player_screen_width,
            ) != self.current_frame.shape[:2]:
                self.controller.step(
                    {
                        "action": "ChangeResolution",
                        "x": self._start_player_screen_width,
                        "y": self._start_player_screen_height,
                    }
                )

            self._started = True
            self.reset(scene_name=scene_name, move_mag=move_mag, **kwargs)

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

        if self.object_open_speed != 1.0:
            self.controller.step(
                {"action": "ChangeOpenSpeed", "x": self.object_open_speed}
            )

        self._initially_reachable_points = None
        self._initially_reachable_points_set = None
        self.controller.step({"action": "GetReachablePositions"})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            get_logger().warning(
                "Error when getting reachable points: {}".format(
                    self.controller.last_event.metadata["errorMessage"]
                )
            )
        self._initially_reachable_points = self.last_action_return

        self.list_of_actions_so_far = []