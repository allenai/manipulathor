"""Utility classes and functions for sensory inputs used by the models."""
import datetime
import os
import random
from typing import Any

import cv2
import gym
import numpy as np
import torch

# from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.embodiedai.sensors.vision_sensors import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from ithor_arm.ithor_arm_constants import DONT_USE_ALL_POSSIBLE_OBJECTS_EVER
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.thor_category_names import thor_possible_objects

class FancyNoisyObjectMaskWLabels(Sensor):
    def __init__(self, type: str,noise, height, width,  uuid: str = "object_mask", distance_thr: float = -1, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.distance_thr = distance_thr
        space_dict = {
            "mask": observation_space,
            'is_real_mask': gym.spaces.Box( low=0, high=1, shape=(1,), dtype=np.bool),
        }
        observation_space = gym.spaces.Dict(space_dict)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        ForkedPdb().set_trace()

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
                if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20: #TODO objects that are smaller than this many pixels should be removed. High chance all spatulas will be removed
                    mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        if len(env.controller.last_event.instance_masks) == 0:
            fake_mask = np.zeros(env.controller.last_event.frame[:,:,0].shape)
        else:
            fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
        fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
        fake_mask = add_mask_noise(result, fake_mask, noise=self.noise)
        current_shape = fake_mask.shape
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = fake_mask
        else:
            resized_mask = cv2.resize(fake_mask, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow

        return {k: map_dict[k] for k in self.observation_space.spaces.keys()}
        return resized_mask
