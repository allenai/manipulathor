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
    MOVE_THR, PICKUP, DONE, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, SET_OF_ALL_AGENT_ACTIONS,
)
from manipulathor_utils.debugger_util import ForkedPdb


class ExploreEnvironment(IThorEnvironment):
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
            docker_enabled: bool = False,
            local_thor_build: Optional[str] = None,
            visibility_distance: float = VISIBILITY_DISTANCE,
            fov: float = FOV,
            player_screen_width: int = 224,
            player_screen_height: int = 224,
            quality: str = "Very Low",
            restrict_to_initially_reachable_points: bool = False,
            make_agents_visible: bool = True,
            object_open_speed: float = 1.0,
            simplify_physics: bool = False,
            verbose: bool = False,
            env_args=None,
    ) -> None:
        """Initializer.

        # Parameters

        x_display : The x display into which to launch ai2thor (possibly necessarily if you are running on a server
            without an attached display).
        docker_enabled : Whether or not to run thor in a docker container (useful on a server without an attached
            display so that you don't have to start an x display).
        local_thor_build : The path to a local build of ai2thor. This is probably not necessary for your use case
            and can be safely ignored.
        visibility_distance : The distance (in meters) at which objects, in the viewport of the agent,
            are considered visible by ai2thor and will have their "visible" flag be set to `True` in the metadata.
        fov : The agent's camera's field of view.
        width : The width resolution (in pixels) of the images returned by ai2thor.
        height : The height resolution (in pixels) of the images returned by ai2thor.
        quality : The quality at which to render. Possible quality settings can be found in
            `ai2thor._quality_settings.QUALITY_SETTINGS`.
        restrict_to_initially_reachable_points : Whether or not to restrict the agent to locations in ai2thor
            that were found to be (initially) reachable by the agent (i.e. reachable by the agent after resetting
            the scene). This can be useful if you want to ensure there are only a fixed set of locations where the
            agent can go.
        make_agents_visible : Whether or not the agent should be visible. Most noticable when there are multiple agents
            or when quality settings are high so that the agent casts a shadow.
        object_open_speed : How quickly objects should be opened. High speeds mean faster simulation but also mean
            that opening objects have a lot of kinetic energy and can, possibly, knock other objects away.
        simplify_physics : Whether or not to simplify physics when applicable. Currently this only simplies object
            interactions when opening drawers (when simplified, objects within a drawer do not slide around on
            their own when the drawer is opened or closed, instead they are effectively glued down).
        """

        self._start_player_screen_width = player_screen_width
        self._start_player_screen_height = player_screen_height
        self._local_thor_build = local_thor_build
        self.x_display = x_display
        self.controller: Optional[Controller] = None
        self._started = False
        self._quality = quality
        self._verbose = verbose
        self.env_args = env_args

        self._initially_reachable_points: Optional[List[Dict]] = None
        self._initially_reachable_points_set: Optional[Set[Tuple[float, float]]] = None
        self._move_mag: Optional[float] = None
        self._grid_size: Optional[float] = None
        self._visibility_distance = visibility_distance

        self._fov = fov

        self.restrict_to_initially_reachable_points = (
            restrict_to_initially_reachable_points
        )
        self.make_agents_visible = make_agents_visible
        self.object_open_speed = object_open_speed
        self._always_return_visible_range = False
        self.simplify_physics = simplify_physics

        self.start(None)

        # noinspection PyTypeHints
        self.controller.docker_enabled = docker_enabled  # type: ignore

    def create_controller(self):
        # NOTE @samir add your environment initialization here
        controller = Controller(
            commit_id="6f13532966080a051127167c6eb2117e47d96f3a",
            height=224,
            width=224,
            renderInstanceSegmentation=False,
            renderDepthImage=False,
            quality="Very Low")

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
        self.memory_frames = []

        if scene_name is None:
            scene_name = self.controller.last_event.metadata["sceneName"]
        # self.reset_init_params()#**kwargs) removing this fixes one of the crashing problem

        # to solve the crash issue
        # why do we still have this crashing problem?
        try:
            self.controller.reset(scene_name)
        except Exception as e:
            print("RESETTING THE SCENE,", scene_name, 'because of', str(e))
            # NOTE @samir
            self.controller = ai2thor.controller.Controller(
                controller = Controller(
                commit_id="6f13532966080a051127167c6eb2117e47d96f3a",
                height=224,
                width=224,
                renderInstanceSegmentation=False,
                renderDepthImage=False,
                quality="Very Low")
            )
            self.controller.reset(scene_name)

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

    def randomize_agent_location(
            self, seed: int = None, partial_position: Optional[Dict[str, float]] = None
    ) -> Dict:
        raise Exception("not used")

    def get_object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        for o in self.last_event.metadata["objects"]:
            if o["objectId"] == object_id:
                o["position"] = self.correct_nan_inf(o["position"], "obj id")
                return o
        return None

    def get_current_object_locations(self):
        obj_loc_dict = {}
        metadata = self.controller.last_event.metadata["objects"]
        for o in metadata:
            obj_loc_dict[o["objectId"]] = dict(
                position=o["position"], rotation=o["rotation"]
            )
        return copy.deepcopy(obj_loc_dict)

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
        if action == MOVE_AHEAD:
            # NOTE @samir add action you want
            action_dict['action'] = 'MoveAhead'
            pass
        if action == ROTATE_RIGHT:
            # NOTE @samir add action you want
            action_dict['action'] = 'RotateRight'
            pass
        if action == ROTATE_LEFT:
            # NOTE @samir add action you want
            action_dict['action'] = 'RotateLeft'
            pass

        sr = self.controller.step(action_dict)
        self.list_of_actions_so_far.append(action_dict)

        if self.restrict_to_initially_reachable_points:
            self._snap_agent_to_initially_reachable()

        if skip_render:
            assert last_frame is not None
            self.last_event.frame = last_frame

        return sr
