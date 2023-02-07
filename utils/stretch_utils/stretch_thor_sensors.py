import copy
import datetime
import os
import random

import torch
from typing import Any, Union, Optional

import gym
import numpy as np
# from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, Sensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms

from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
# from ithor_arm.bring_object_sensors import NoisyObjectMask, add_mask_noise
# from ithor_arm.ithor_arm_environment import StretchManipulaTHOREnvironment
# from ithor_arm.ithor_arm_sensors import DepthSensorThor
# from ithor_arm.near_deadline_sensors import calc_world_coordinates
from utils.calculation_utils import calc_world_coordinates

from manipulathor_utils.debugger_util import ForkedPdb
from utils.noise_in_motion_util import squeeze_bool_mask
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from utils.stretch_utils.stretch_sim2real_utils import intel_reshape




class DepthSensorStretchNav(
    DepthSensor[
        Union[StretchManipulaTHOREnvironment],
        Union[Task[StretchManipulaTHOREnvironment]],
    ]
):
    """Sensor for Depth images in THOR.

    Returns from a running StretchManipulaTHOREnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: StretchManipulaTHOREnvironment, task: Optional[Task]) -> np.ndarray:

        return env.nav_depth


class DepthSensorStretchManip(
    DepthSensor[
        Union[StretchManipulaTHOREnvironment],
        Union[Task[StretchManipulaTHOREnvironment]],
    ]
):
    """Sensor for Depth images in THOR.

    Returns from a running StretchManipulaTHOREnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: StretchManipulaTHOREnvironment, task: Optional[Task]) -> np.ndarray:

        return env.manip_depth




class RGBSensorStretchManip(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running StretchManipulaTHOREnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: StretchManipulaTHOREnvironment, task: Optional[Task]) -> np.ndarray:
        return env.manip_rgb


class RGBSensorStretchNav(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running StretchManipulaTHOREnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: StretchManipulaTHOREnvironment, task: Optional[Task]) -> np.ndarray:

        return env.nav_rgb

def get_new_transformation():#TODO fix these values? in each iteration
    return transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)], p=0.8),
        transforms.ToTensor(),
    ])
def apply_transformation(rgb, transformation):
    rgb = transformation(rgb)
    rgb = rgb.permute(1, 2, 0)
    rgb = rgb * 255.
    return rgb.numpy().astype(np.uint8)

class RGBSensorStretchNavwJitter(
    RGBSensorThor
):
    def frame_from_env(self, env: StretchManipulaTHOREnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = (env.nav_rgb_raw.copy()) #TODO this needs to get raw images and thne apply the changes
        if task.num_steps_taken() == 0:
            self.jitter_function = get_new_transformation()
        rgb = apply_transformation(rgb, self.jitter_function)
        return intel_reshape(rgb, camera_needs_rotation=True) #TODO this is double


def check_validity(depth_frame, controller, scene_number=''):
    if np.any(depth_frame != depth_frame) or np.any(np.isinf(depth_frame)):
        print('OH THERE IS SOMETHING OFF WITH THIS FRAME', scene_number, depth_frame.min(), depth_frame.max(), depth_frame.mean(), np.linalg.norm(depth_frame))
        dir_to_save = 'experiment_output/depth_errors'
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        timestamp = timestamp + '_scene_' + str(scene_number) + '.png'
        os.makedirs(dir_to_save, exist_ok=True)

        frame_to_save = np.concatenate([controller.last_event.frame, normalize_depth(controller.last_event.depth_frame), controller.last_event.third_party_camera_frames[0], normalize_depth(controller.last_event.third_party_depth_frames[0]),normalize_depth(depth_frame)], axis=1)
        plt.imsave(os.path.join(dir_to_save, timestamp), np.clip(frame_to_save / 255., 0, 1))

def normalize_depth(depth):
    return depth[:,:,np.newaxis].repeat(3,axis=2) * (255. / depth.max())

class NavCameraNoisyObjectMask(Sensor):
    def __init__(self, type: str,noise, height, width,  uuid: str = "object_mask_nav", distance_thr: float = -1, only_close_big_masks=False, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.distance_thr = distance_thr
        self.only_close_big_masks = only_close_big_masks
        super().__init__(**prepare_locals_for_super(locals()))
        assert self.noise == 0

    def get_observation(
            self, env: StretchManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        else:
            raise Exception('Not implemented', self.type)

        target_object_id = task.task_info[info_to_search]
        all_visible_masks = env.controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]

            if self.distance_thr > 0:

                agent_location = env.get_agent_location()
                object_location = env.get_object_by_id(target_object_id)['position']
                current_agent_distance_to_obj = sum([(object_location[k] - agent_location[k])**2 for k in ['x', 'z']]) ** 0.5

                if self.only_close_big_masks:
                    if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20: # objects that are smaller than this many pixels should be removed. High chance all spatulas will be removed

                        mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))

        assert self.noise == 0
        current_shape = result.shape
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = result
        else:
            resized_mask = cv2.resize(result, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow
        return intel_reshape(resized_mask)

class ManipCameraNoisyObjectMask(Sensor):
    def __init__(self, type: str,noise, height, width,  uuid: str = "object_mask_manip", distance_thr: float = -1, only_close_big_masks=False, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.distance_thr = distance_thr
        self.only_close_big_masks = only_close_big_masks
        super().__init__(**prepare_locals_for_super(locals()))
        assert self.noise == 0

    def get_observation(
            self, env: StretchManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        else:
            raise Exception('Not implemented', self.type)

        target_object_id = task.task_info[info_to_search]
        all_visible_masks = env.controller.last_event.third_party_instance_masks[0]
        if len(env.controller.last_event.third_party_instance_masks) != 1:
            print('Warning multiple cameras')
        # assert len(env.controller.last_event.third_party_instance_masks) == 1
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]

            if self.distance_thr > 0:

                agent_location = env.get_agent_location()
                object_location = env.get_object_by_id(target_object_id)['position']
                current_agent_distance_to_obj = sum([(object_location[k] - agent_location[k])**2 for k in ['x', 'z']]) ** 0.5
                if self.only_close_big_masks:
                    if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20: # objects that are smaller than this many pixels should be removed. High chance all spatulas will be removed

                        mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        mask_frame = (np.expand_dims(mask_frame.astype(np.float),axis=-1))

        current_shape = mask_frame.shape
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = mask_frame
        else:
            resized_mask = cv2.resize(mask_frame, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow

        return kinect_reshape(resized_mask)

class StretchPickedUpObjSensor(Sensor):
    def __init__(self, uuid: str = "pickedup_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: StretchManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return task.object_picked_up