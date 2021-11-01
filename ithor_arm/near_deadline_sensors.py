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
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import depth_frame_to_world_space_xyz, project_point_cloud_to_map
from allenact.embodiedai.sensors.vision_sensors import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from ithor_arm.arm_calculation_utils import convert_world_to_agent_coordinate, diff_position, convert_state_to_tensor
from ithor_arm.bring_object_sensors import add_mask_noise
from ithor_arm.ithor_arm_constants import DONT_USE_ALL_POSSIBLE_OBJECTS_EVER
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.pointcloud_sensors import rotate_points_to_agent, KianaReachableBoundsTHORSensor
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
                if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20:
                    mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        real_mask = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        if len(env.controller.last_event.instance_masks) == 0:
            fake_mask = np.zeros(env.controller.last_event.frame[:,:,0].shape)
        else:
            fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
        fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
        fake_mask, is_real_mask = add_mask_noise(real_mask, fake_mask, noise=self.noise)


        return {'mask': fake_mask, 'is_real_mask':torch.tensor(is_real_mask).long()}

class PointNavEmulatorSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor,  uuid: str = "point_nav_emul", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        uuid = '{}_{}'.format(uuid, type)
        self.pointnav_history_aggr = None
        self.map_range_sensor = KianaReachableBoundsTHORSensor(margin=1.0)
        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 #TODO is this good enough?
        self.device = torch.device("cpu")
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        mask = self.mask_sensor.get_observation(env, task, *args, **kwargs).astype(bool).squeeze(-1)
        depth_frame = env.controller.last_event.depth_frame.copy()
        depth_frame[~mask] = -1
        assert mask.shape == depth_frame.shape

        if task.num_steps_taken() == 0:
            # xyz_ranges_dict = self.map_range_sensor.get_observation(env=env, task=task)
            # self.min_xyz = np.array(
            #     [
            #         xyz_ranges_dict["x_range"][0],
            #         0, # TODO xyz_ranges_dict["y_range"][0],
            #         xyz_ranges_dict["z_range"][0],
            #     ]
            # )
            self.pointnav_history_aggr = []
        #TODO why?
        self.min_xyz = np.zeros((3))

        metadata = env.controller.last_event.metadata
        camera_xyz = np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]])
        camera_rotation=metadata["agent"]["rotation"]["y"]
        camera_horizon=metadata["agent"]["cameraHorizon"]
        fov = metadata['fov']
        arm_state = env.get_absolute_hand_state()
        # TODO should we use this for all of them?
        # arm_state = dict(position=env.controller.last_event.metadata['arm']['handSphereCenter'], rotation=dict(x=0, y=0,z=0))

        if mask.sum() == 0:
            return self.average_so_far(camera_xyz, camera_rotation, arm_state)

        world_space_point_cloud = calc_world_coordinates(self.min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, self.device, depth_frame)


        valid_points = (world_space_point_cloud == world_space_point_cloud).sum(dim=-1) == 3
        point_in_world = world_space_point_cloud[valid_points]
        middle_of_object = point_in_world.mean(dim=0)
        self.pointnav_history_aggr.append((middle_of_object.cpu(), len(point_in_world)))

        return self.average_so_far(camera_xyz, camera_rotation, arm_state)


    def average_so_far(self, camera_xyz, camera_rotation, arm_state):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            total_sum = [k * v for k,v in self.pointnav_history_aggr]
            total_sum = sum(total_sum)
            total_count = sum([v for k,v in self.pointnav_history_aggr])
            midpoint = total_sum / total_count
            self.pointnav_history_aggr = [(midpoint, total_count)]
            # agent_centric_middle_of_object = rotate_to_agent(midpoint, self.device, camera_xyz, camera_rotation)
            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            arm_state_agent_coord = convert_world_to_agent_coordinate(arm_state, agent_state)
            distance_in_agent_coord = dict(x=arm_state_agent_coord['position']['x'] - midpoint_agent_coord['position']['x'],y=arm_state_agent_coord['position']['y'] - midpoint_agent_coord['position']['y'],z=arm_state_agent_coord['position']['z'] - midpoint_agent_coord['position']['z'])

            # distance_to_obj = dict(x=arm_state['position']['x'] - midpoint[0],y=arm_state['position']['y'] - midpoint[1],z=arm_state['position']['z'] - midpoint[2])
            # env.get_object_by_id('Pan|+01.38|+01.74|+00.39')
            # env.get_absolute_hand_state()
            # distance_to_obj = dict(position=distance_to_obj, rotation=dict(x=0,y=0,z=0))
            # agent_centric_middle_of_object = convert_world_to_agent_coordinate(distance_to_obj, agent_state)
            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])
            #TODO this is too big! we basically are saying that we have arm supervision as well

            #TODO IS THIS REALLY THE PROBLEM???
            agent_centric_middle_of_object = agent_centric_middle_of_object.abs()
            return agent_centric_middle_of_object

def not_working_rotate_to_agent(middle_of_object, device, camera_xyz, camera_rotation):
    recentered_point_cloud = middle_of_object - (torch.FloatTensor([1.0, 0.0, 1.0]).to(device) * camera_xyz).float().reshape((1, 1, 3))
    # Rotate the cloud so that positive-z is the direction the agent is looking
    theta = (np.pi * camera_rotation / 180)  # No negative since THOR rotations are already backwards
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_transform = torch.FloatTensor([[cos_theta, 0, -sin_theta],[0, 1, 0], [sin_theta, 0, cos_theta],]).to(device)
    rotated_point_cloud = recentered_point_cloud @ rotation_transform.T
    # xoffset = (map_size_in_cm / 100) / 2
    # agent_centric_point_cloud = rotated_point_cloud + torch.FloatTensor([xoffset, 0, 0]).to(device)
    return rotated_point_cloud.squeeze(0).squeeze(0)

def calc_world_coordinates(min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, device, depth_frame):
    with torch.no_grad():
        camera_xyz = (
            torch.from_numpy(camera_xyz - min_xyz).float().to(device)
        )

        depth_frame = torch.from_numpy(depth_frame).to(device)
        depth_frame[depth_frame == -1] = np.NaN
        world_space_point_cloud = depth_frame_to_world_space_xyz(
            depth_frame=depth_frame,
            camera_world_xyz=camera_xyz,
            rotation=camera_rotation,
            horizon=camera_horizon,
            fov=fov,
        )
        return world_space_point_cloud


class TempRealArmpointNav(Sensor):

    def __init__(self, type: str, uuid: str = "point_nav_real", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        goal_obj_id = task.task_info[info_to_search]
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
