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

# from legacy.from_phone_to_sim.more_optimized import get_point_cloud
# from legacy.from_phone_to_sim.thor_frames_to_pointcloud import frames_to_world_points, world_points_to_pointcloud
from utils.from_phone_to_sim.thor_frames_to_pointcloud import frames_to_world_points, world_points_to_pointcloud, save_pointcloud_to_file
from utils.manipulathor_data_loader_utils import get_random_query_image_from_img_adr, get_random_query_feature_from_img_adr


class RelativeArmDistanceToGoal(Sensor):
    def __init__(self, uuid: str = "relative_arm_dist", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        is_object_picked_up = task.object_picked_up
        if not is_object_picked_up: #TODO is this the updated one or one step before?
            distance = task.arm_distance_from_obj()
        else:
            distance = task.obj_distance_from_goal()
        return distance


class PreviousActionTaken(Sensor):
    def __init__(self,  uuid: str = "previous_action_taken", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        last_action = task._last_action_str
        action_list = task._actions
        result = torch.zeros(len(action_list))
        if last_action != None:
            result[action_list.index(last_action)] = 1
        return result.bool()


class IsGoalObjectVisible(Sensor):
    def __init__(self,  uuid: str = "is_goal_object_visible", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if not task.object_picked_up:
            result = task.source_observed_reward
        else:
            result = task.goal_observed_reward
        return torch.tensor(result).bool()


class NoGripperRGBSensorThor(RGBSensorThor):
    def frame_from_env(
            self, env: IThorEnvironment, task: Task[IThorEnvironment]
    ) -> np.ndarray:  # type:ignore
        env.controller.step('ToggleMagnetVisibility')
        frame = env.current_frame.copy()
        env.controller.step('ToggleMagnetVisibility')
        return frame

class CategorySampleSensor(Sensor):
    def __init__(self, type: str, uuid: str = "category_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        feature_name = 'object_query'
        if self.type == 'source':
            info_to_search = 'source_' + feature_name
        elif self.type == 'destination':
            info_to_search = 'goal_' + feature_name
        else:
            raise Exception('Not implemented', self.type)
        # image_adr = task.task_info[info_to_search]
        # image = get_random_query_image_from_img_adr(image_adr)
        image = task.task_info[info_to_search]
        return image


class CategoryFeatureSampleSensor(Sensor):
    def __init__(self, type: str, uuid: str = "category_object_feature", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        # feature_name = 'object_query_file_name'
        feature_name = 'object_query_feature'
        if self.type == 'source':
            info_to_search = 'source_' + feature_name
        elif self.type == 'destination':
            info_to_search = 'goal_' + feature_name
        else:
            raise Exception('Not implemented', self.type)
        # image_adr = task.task_info[info_to_search]
        # feature = get_random_query_feature_from_img_adr(image_adr)
        feature = task.task_info[info_to_search]
        return feature



class NoisyObjectMask(Sensor):
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
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
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
                # if current_agent_distance_to_obj > self.distance_thr:# or mask_frame.sum() < 20: #TODO objects that are smaller than this many pixels should be removed. High chance all spatulas will be removed
                #     mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        if len(env.controller.last_event.instance_masks) == 0:
            fake_mask = np.zeros(env.controller.last_event.frame[:,:,0].shape)
        else:
            fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
        fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
        fake_mask, is_real_mask = add_mask_noise(result, fake_mask, noise=self.noise)
        current_shape = fake_mask.shape
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = fake_mask
        else:
            resized_mask = cv2.resize(fake_mask, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow
        return resized_mask

class NoMaskSensor(NoisyObjectMask):
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        return result

class NoisyObjectRegion(NoisyObjectMask):
    def __init__(self, type: str,noise, region_size,height, width,  uuid: str = "object_mask", distance_thr: float = -1, **kwargs: Any):
        super().__init__(**prepare_locals_for_super(locals()))
        self.region_size = region_size
        assert self.region_size == 14, 'the following need to be changed'
        # self.avgpool = torch.nn.AvgPool2d(16, stride=16)

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        mask = super(type(self), self).get_observation(env, task, *args, **kwargs)


        region = cv2.resize(mask, (self.region_size, self.region_size))
        region = (region > 0.1).astype(float).reshape(self.region_size, self.region_size, 1)
        assert self.region_size == 14, 'the folliowing number wont work'
        number_of_repeat = 16
        region = region.repeat(number_of_repeat, axis=0).repeat(number_of_repeat, axis=1)

        #
        # with torch.no_grad():  this takes  forever
        #     region = torch.tensor(mask).permute(2, 0, 1)
        #     region = self.avgpool(region)
        #     region = (region > 0.1).float()
        #     region = torch.nn.functional.interpolate(region.unsqueeze(0), (self.width, self.height)).squeeze(0).permute(1,2,0)

        return region


def add_mask_noise(real_mask, fake_mask, noise):
    TURN_OFF_RATE = noise
    REMOVE_RATE = noise
    REPLACE_WITH_FAKE = noise

    result = real_mask.copy()

    random_prob = random.random()
    if random_prob < REMOVE_RATE:
        result[:] = 0.
        is_real_mask = False
    elif random_prob < REMOVE_RATE + REPLACE_WITH_FAKE: #TODO I think this is too much, think of all the frames that we don't actually see the object but this is true
        result = fake_mask
        is_real_mask = False
    elif random_prob < REMOVE_RATE + REPLACE_WITH_FAKE + TURN_OFF_RATE:
        w, h, d = result.shape
        mask = np.random.rand(w, h, d)
        mask = mask < TURN_OFF_RATE
        mask = mask & (result == 1)
        result[mask] = 0
        is_real_mask = True
    else:
        is_real_mask = True

    masks_are_changed_but_still_similar = (result != real_mask).sum() == 0 and not is_real_mask
    if masks_are_changed_but_still_similar:
        is_real_mask = True

    return result, is_real_mask



class TempObjectCategorySensor(Sensor):
    def __init__(self, type: str, uuid: str = "temp_category_code", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        uuid = '{}_{}'.format(uuid, type)
        self.type = type
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        else:
            raise Exception('Not implemented', self.type)
        object_type = task.task_info[info_to_search].split('|')[0]
        object_type_categ_ind = DONT_USE_ALL_POSSIBLE_OBJECTS_EVER.index(object_type)
        return torch.tensor(object_type_categ_ind)
class TempEpisodeNumber(Sensor):
    def __init__(self, uuid: str = "temp_episode_number", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return torch.tensor(task.task_info['episode_number'])

class TempAllMasksSensor(Sensor):
    def __init__(self, uuid: str = "all_masks_sensor", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        result = torch.zeros((224,224))
        result[:] = -1
        for (obj, mask) in env.controller.last_event.instance_masks.items():
            object_type = obj.split('|')[0]
            if object_type not in DONT_USE_ALL_POSSIBLE_OBJECTS_EVER:
                continue
            object_type_categ_ind = DONT_USE_ALL_POSSIBLE_OBJECTS_EVER.index(object_type)
            result[mask] = object_type_categ_ind
        return result

class PointCloudMemory(Sensor):
    def __init__(self,memory_size, mask_generator, uuid: str = "point_cloud", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.memory_size = memory_size
        self.mask_generator = mask_generator
        super().__init__(**prepare_locals_for_super(locals()))
        self.all_masks = []



    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        # assert self.memory_size == env.MEMORY_SIZE

        if len(env.memory_frames) == 0:
            return 10 #LATER_TODO
        mask = self.mask_generator.get_observation(env, task, *args, **kwargs)
        self.all_masks.append(mask)
        frames = [k['rgb'] for k in env.memory_frames]
        depth_frames = [k['depth'] for k in env.memory_frames]
        metadatas = [k['event'] for k in env.memory_frames]



        #LATER_TODO
        if False:
            #option 2
            pc = get_point_cloud(frames, depth_frames, metadatas)
        elif False:
            #option1
            # if random.random() < 1/5.:
            #     save_pointcloud_to_file(pc, os.path.join(dir_to_save, timesmap))
            #     print('saved pointcloud', os.path.join(dir_to_save, timesmap))
            # #LATER_TODO we probably need to convert this back to agent's coordinate frame
            if len(env.memory_frames) > 150:
                print('starting to generate pointcloud')
                xyz, normals, rgb = frames_to_world_points(metadatas, frames, depth_frames)
                pc = world_points_to_pointcloud(xyz, normals, rgb, voxel_size=0.02)
                dir_to_save = 'experiment_output/visualization_pointcloud/'
                os.makedirs(dir_to_save, exist_ok=True)
                timesmap = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.ply")
                ForkedPdb().set_trace()

                save_pointcloud_to_file(pc, os.path.join(dir_to_save, timesmap))
        else:
            if len(env.memory_frames) % 100 == 0 or task.object_picked_up:
                def generate_and_save_pointcloud():
                    print('starting to generate pointcloud')
                    for i in range(len(env.memory_frames)):
                        mask = self.all_masks[i].squeeze(-1).astype(bool)
                        env.memory_frames[i]['depth'][~mask] = float('nan')

                        # env.memory_frames[i]['rgb'][:,:,0] = env.memory_frames[i]['rgb'].mean(-1)
                        # env.memory_frames[i]['rgb'][:,:,1] = env.memory_frames[i]['rgb'].mean(-1)
                        # env.memory_frames[i]['rgb'][:,:,2] = env.memory_frames[i]['rgb'].mean(-1)
                        gray_scale = env.memory_frames[i]['rgb'].mean(-1)

                        # env.memory_frames[i]['rgb'][:,:, 0] = gray_scale
                        # env.memory_frames[i]['rgb'][:,:,1] = gray_scale
                        # env.memory_frames[i]['rgb'][:,:,2] = gray_scale
                        env.memory_frames[i]['rgb'][:,:, 0] = 0
                        env.memory_frames[i]['rgb'][:,:,1] = 0
                        env.memory_frames[i]['rgb'][:,:,2] = 0
                    frames = [k['rgb'] for k in env.memory_frames]
                    depth_frames = [k['depth'] for k in env.memory_frames]
                    metadatas = [k['event'] for k in env.memory_frames]
                    xyz, normals, rgb = frames_to_world_points(metadatas, frames, depth_frames)
                    pc = world_points_to_pointcloud(xyz, normals, rgb, voxel_size=0.02)
                    dir_to_save = 'experiment_output/visualization_pointcloud/'
                    os.makedirs(dir_to_save, exist_ok=True)
                    timesmap = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.ply")
                    save_pointcloud_to_file(pc, os.path.join(dir_to_save, timesmap))
                ForkedPdb().set_trace()

                # save_pointcloud_to_file(pc, os.path.join(dir_to_save, timesmap))
        return 10



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
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any): # maybe we should change this name but since i wanted to use the same model
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


#
# class NoisyTargetLocationMask(Sensor): #
#     def __init__(self, uuid: str = "target_location_mask", **kwargs: Any):
#         observation_space = gym.spaces.Box(
#             low=0, high=1, shape=(1,), dtype=np.float32
#         )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
#         super().__init__(**prepare_locals_for_super(locals()))
#
#
#     def get_observation(
#             self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
#     ) -> Any:
#         target_object_id = task.task_info['goal_object_id']
#         all_visible_masks = env.controller.last_event.instance_masks
#         if target_object_id in all_visible_masks:
#             mask_frame = all_visible_masks[target_object_id]
#         else:
#             mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
#
#         result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
#         fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
#         fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
#
#         return add_mask_noise(result, fake_mask)
#
# class NoisyTargetObjectMask(Sensor):
#
#     def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
#         observation_space = gym.spaces.Box(
#             low=0, high=1, shape=(1,), dtype=np.float32
#         )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
#         super().__init__(**prepare_locals_for_super(locals()))
#
#     def get_observation(
#             self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
#     ) -> Any:
#         target_object_id = task.task_info['source_object_id']
#         all_visible_masks = env.controller.last_event.instance_masks
#         if target_object_id in all_visible_masks:
#             mask_frame = all_visible_masks[target_object_id]
#         else:
#             mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)
#
#         result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
#         fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
#         fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
#         return add_mask_noise(result, fake_mask)



class PredictedTargetObjectMask(Sensor): #LATER_TODO if this is too slow then change it
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))
        self.mask_predictor = ConditionalDetectionModel() #LATER_TODO can we have only one for both of target object and location and also save it in one place. this might explode
        self.mask_predictor.eval()
        print('resolve todos?')
        ForkedPdb().set_trace()
        #LATER_TODO load weights into this

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        with torch.no_grad():
            target_object_id = task.task_info['source_object_id'] #LATER_TODO just have one image instead of this
            target_image = get_target_image(target_object_id)
            target_image = mean_subtract_normalize_image(target_image)
            current_image = torch.Tensor(env.controller.last_event.frame.copy()).float()
            current_image = mean_subtract_normalize_image(current_image)
            #LATER_TODO double check beacause the load and the other shit might be different

            #LATER_TODO visualize the outputs
            predictions = self.mask_predictor(dict(rgb=current_image.unsqueeze(0), target_cropped_object=target_image.unsqueeze(0)))
            probs_mask = predictions['object_mask'].squeeze(0) #Remove batch size
            mask = probs_mask.argmax(dim=0).float().unsqueeze(-1) # To add a channel
            return mask


def get_target_image(object_id): #LATER_TODO implement
    return torch.zeros((224,224, 3))

def mean_subtract_normalize_image(image):
    #LATER_TODO implement
    return image.permute(2,0,1)












def add_noise(result):
    TURN_OFF_RATE = 0.05 # TODO good?
    ADD_PIXEL_RATE = 0.0005 # TODO good?
    REMOVE_RATE = 0.1 # TODO what do you think?
    if random.random() < REMOVE_RATE:
        result[:] = 0.
    else:
        w, h, d = result.shape
        mask = np.random.rand(w, h, d)
        mask = mask < TURN_OFF_RATE
        mask = mask & (result == 1)
        result[mask] = 0


        mask = np.random.rand(w, h, d)
        mask = mask < ADD_PIXEL_RATE
        mask = mask & (result == 0)
        result[mask] = 1

    return result

class NoisyTargetLocationBBox(Sensor): #
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
        result = (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))

        return add_noise(result)



class NoisyTargetObjectBBox(Sensor):
    def __init__(self, uuid: str = "target_object_mask", **kwargs: Any):
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
        result = (np.expand_dims(box_as_mask_frame.astype(np.float),axis=-1))

        return add_noise(result)
