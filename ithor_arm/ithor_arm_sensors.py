"""Utility classes and functions for sensory inputs used by the models."""
from typing import Any, Union, Optional

import gym
import numpy as np
from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_utils.debugger_util import ForkedPdb


class DepthSensorThor(
    DepthSensor[
        Union[IThorEnvironment],
        Union[Task[IThorEnvironment]],
    ]
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = (env.controller.last_event.depth_frame.copy())

        # TODO remove
        # if True:
        #     try:
        #         self.depth_dict
        #     except Exception:
        #         self.depth_dict = {
        #             'min': [],
        #             'max': [],
        #             'mean': [],
        #             'norm': [],
        #
        #         }
        #     import torch
        #     depth_frame = torch.Tensor(depth)
        #     self.depth_dict['min'].append(depth_frame.min())
        #     self.depth_dict['max'].append(depth_frame.max())
        #     self.depth_dict['mean'].append(depth_frame.mean())
        #     self.depth_dict['norm'].append(depth_frame.norm())
        #     print('total', len(self.depth_dict['min']),
        #           'min', sum(self.depth_dict['min']) / len(self.depth_dict['min']),
        #           'max', sum(self.depth_dict['max']) / len(self.depth_dict['max']),
        #           'mean', sum(self.depth_dict['mean']) / len(self.depth_dict['mean']),
        #           'norm', sum(self.depth_dict['norm']) / len(self.depth_dict['norm'])
        #           )
        return depth


class NoVisionSensorThor(
    RGBSensor[
        Union[IThorEnvironment],
        Union[Task[IThorEnvironment]],
    ]
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:
        result = env.current_frame.copy()
        result.fill(0)
        return result


class AgentRelativeCurrentObjectStateThorSensor(Sensor):
    def __init__(self, uuid: str = "relative_current_obj_state", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(6,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: IThorEnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        object_id = task.task_info["objectId"]
        current_object_state = env.get_object_by_id(object_id)
        relative_current_obj = convert_world_to_agent_coordinate(
            current_object_state, env.controller.last_event.metadata["agent"]
        )
        result = convert_state_to_tensor(
            dict(
                position=relative_current_obj["position"],
                rotation=relative_current_obj["rotation"],
            )
        )
        return result


class RelativeObjectToGoalSensor(Sensor):
    def __init__(self, uuid: str = "relative_obj_to_goal", **kwargs: Any):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        target_state = task.task_info["target_location"]

        agent_state = env.controller.last_event.metadata["agent"]

        relative_current_obj = convert_world_to_agent_coordinate(
            object_info, agent_state
        )
        relative_goal_state = convert_world_to_agent_coordinate(
            target_state, agent_state
        )
        relative_distance = diff_position(relative_current_obj, relative_goal_state)
        result = convert_state_to_tensor(dict(position=relative_distance))
        return result


class InitialObjectToGoalSensor(Sensor):
    def __init__(self, uuid: str = "initial_obj_to_goal", **kwargs: Any):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        object_source_location = task.task_info['initial_object_location']
        target_state = task.task_info["target_location"]
        agent_state = task.task_info['agent_initial_state']

        relative_current_obj = convert_world_to_agent_coordinate(
            object_source_location, agent_state
        )
        relative_goal_state = convert_world_to_agent_coordinate(
            target_state, agent_state
        )
        relative_distance = diff_position(relative_current_obj, relative_goal_state)
        result = convert_state_to_tensor(dict(position=relative_distance))
        return result


class DistanceObjectToGoalSensor(Sensor):
    def __init__(self, uuid: str = "distance_obj_to_goal", **kwargs: Any):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        target_state = task.task_info["target_location"]

        agent_state = env.controller.last_event.metadata["agent"]

        relative_current_obj = convert_world_to_agent_coordinate(
            object_info, agent_state
        )
        relative_goal_state = convert_world_to_agent_coordinate(
            target_state, agent_state
        )
        relative_distance = diff_position(relative_current_obj, relative_goal_state)
        result = convert_state_to_tensor(dict(position=relative_distance))

        result = ((result ** 2).sum()**0.5).view(1)
        return result


class RelativeAgentArmToObjectSensor(Sensor):
    def __init__(self, uuid: str = "relative_agent_arm_to_obj", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        hand_state = env.get_absolute_hand_state()

        relative_goal_obj = convert_world_to_agent_coordinate(
            object_info, env.controller.last_event.metadata["agent"]
        )
        relative_hand_state = convert_world_to_agent_coordinate(
            hand_state, env.controller.last_event.metadata["agent"]
        )
        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = convert_state_to_tensor(dict(position=relative_distance))

        return result


class InitialAgentArmToObjectSensor(Sensor):
    def __init__(self, uuid: str = "initial_agent_arm_to_obj", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        object_source_location = task.task_info['initial_object_location']
        initial_hand_state = task.task_info['initial_hand_state']

        relative_goal_obj = convert_world_to_agent_coordinate(
            object_source_location, env.controller.last_event.metadata["agent"]
        )
        relative_hand_state = convert_world_to_agent_coordinate(
            initial_hand_state, env.controller.last_event.metadata["agent"]
        )
        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = convert_state_to_tensor(dict(position=relative_distance))

        return result

class DistanceAgentArmToObjectSensor(Sensor):
    def __init__(self, uuid: str = "distance_agent_arm_to_obj", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        hand_state = env.get_absolute_hand_state()

        relative_goal_obj = convert_world_to_agent_coordinate(
            object_info, env.controller.last_event.metadata["agent"]
        )
        relative_hand_state = convert_world_to_agent_coordinate(
            hand_state, env.controller.last_event.metadata["agent"]
        )
        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = convert_state_to_tensor(dict(position=relative_distance))

        result = ((result ** 2).sum()**0.5).view(1)
        return result


class PickedUpObjSensor(Sensor):
    def __init__(self, uuid: str = "pickedup_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return task.object_picked_up
