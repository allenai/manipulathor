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
from scripts.thor_category_names import thor_possible_objects


class DestinationObjectSensor(Sensor):
    def __init__(self, uuid: str = "destination_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return task.task_info['goal_object_id'].split('|')[0]
class TargetObjectMask(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        return (np.expand_dims(mask_frame.astype(np.float),axis=-1))



class TargetObjectBBox(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any): #TODO maybe we should change this name but since i wanted to use the same model
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.
        return (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))



class TargetObjectType(Sensor):
    def __init__(self, uuid: str = "target_object_type", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['source_object_id']
        target_object_type = target_object_id.split('|')[0]
        return thor_possible_objects.index(target_object_type)

class RawRGBSensorThor(Sensor):
    def __init__(self, uuid: str = "raw_rgb_lowres", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return env.current_frame.copy()



class TargetLocationMask(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        return (np.expand_dims(mask_frame.astype(np.float),axis=-1))

class TargetLocationType(Sensor): #
    def __init__(self, uuid: str = "target_location_type", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']
        target_location_type = target_object_id.split('|')[0]
        return thor_possible_objects.index(target_location_type)


class TargetLocationBBox(Sensor): #
    def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        target_object_id = task.task_info['goal_object_id']

        all_visible_masks = env.controller.last_event.instance_detections2D
        box_as_mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        if target_object_id in all_visible_masks:
            x1, y1, x2, y2 = all_visible_masks[target_object_id]
            box_as_mask_frame[y1:y2, x1:x2] = 1.

        return (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))

