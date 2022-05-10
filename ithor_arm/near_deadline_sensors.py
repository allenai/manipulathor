"""Utility classes and functions for sensory inputs used by the models."""
import copy
import datetime
import math
import os
import platform
import random
from typing import Any, Optional

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


from torch.distributions.utils import lazy_property

from ithor_arm.arm_calculation_utils import convert_world_to_agent_coordinate, diff_position, convert_state_to_tensor
from ithor_arm.bring_object_sensors import add_mask_noise
from ithor_arm.ithor_arm_constants import DONT_USE_ALL_POSSIBLE_OBJECTS_EVER
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_sensors import DepthSensorThor
from ithor_arm.pointcloud_sensors import rotate_points_to_agent, KianaReachableBoundsTHORSensor
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.thor_category_names import thor_possible_objects
from utils.calculation_utils import calc_world_coordinates
from utils.klemens_constants import OMNI_CATEGORIES, OMNI_TO_ITHOR, ITHOR_TO_OMNI

from utils.noise_depth_util_files.sim_depth import RedwoodDepthNoise
from utils.noise_from_habitat import ControllerNoiseModel, MotionNoiseModel, _TruncatedMultivariateGaussian
from utils.noise_in_motion_util import NoiseInMotion, squeeze_bool_mask, tensor_from_dict


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

    def __init__(self, type: str, mask_sensor:Sensor,  uuid: str = "point_nav_emul", noise = 0, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        # self.pointnav_history_aggr = None
        # self.map_range_sensor = KianaReachableBoundsTHORSensor(margin=1.0)
        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")
        self.min_xyz = np.zeros((3))
        # self.real_prev_location = None
        # self.belief_prev_location = None
        self.noise_mode = ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.074, 0.036], [0.019, 0.033]),
                _TruncatedMultivariateGaussian([0.189], [0.038]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.003], [0.0, 0.002]),
                _TruncatedMultivariateGaussian([0.219], [0.019]),
            ),
        )

        # TODO remove
        # def get_samples(num_samples):
        #     samples = [self.noise_mode.linear_motion.linear.sample() for _ in range(num_samples)]
        #     samples = np.array(samples)
        #     return samples
        # samples = get_samples(1000)
        # np.absolute(samples * 2).mean(axis=0)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_accurate_locations(self, env):
        metadata = copy.deepcopy(env.controller.last_event.metadata)
        camera_xyz = np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]])
        camera_rotation=metadata["agent"]["rotation"]["y"]
        camera_horizon=metadata["agent"]["cameraHorizon"]
        arm_state = env.get_absolute_hand_state()
        return dict(camera_xyz=camera_xyz, camera_rotation=camera_rotation, camera_horizon=camera_horizon, arm_state=arm_state)


    def add_translation_noise(self, change_in_xyz, prev_location):

        if np.abs(change_in_xyz).sum() > 0:

            noise_value_x, noise_value_z = self.noise_mode.linear_motion.linear.sample() * 0.01 * self.noise #to convert to meters
            new_change_in_xyz = change_in_xyz.copy()
            new_change_in_xyz[0] += noise_value_x
            new_change_in_xyz[2] += noise_value_z
            real_rotation = self.real_prev_location['camera_rotation']
            belief_rotation = self.belief_prev_location['camera_rotation']
            diff_in_rotation = math.radians(belief_rotation - real_rotation)
            # 𝑥2=cos𝛽𝑥1−sin𝛽𝑦1
            # 𝑦2=sin𝛽𝑥1+cos𝛽𝑦1
            new_location = prev_location.copy()
            x = math.cos(diff_in_rotation) * new_change_in_xyz[0] - math.sin(diff_in_rotation) * new_change_in_xyz[2]
            z = math.sin(diff_in_rotation) * new_change_in_xyz[0] + math.cos(diff_in_rotation) * new_change_in_xyz[2]
            new_location[0] += x
            new_location[2] += z
        else:
            new_location = prev_location + change_in_xyz
        return new_location
    def rotate_x_z_around_center(self, x, z, rotation):

        new_x = math.cos(rotation) * x - math.sin(rotation) * z
        new_z = math.sin(rotation) * x + math.cos(rotation) * z

        return new_x, new_z
    def add_rotation_noise(self, change_in_rotation, prev_rotation):
        new_rotation = prev_rotation + change_in_rotation

        if change_in_rotation > 0:
            noise_in_rotation = self.noise_mode.rotational_motion.rotation.sample().item()
            new_rotation += noise_in_rotation
        return new_rotation


    def get_agent_localizations(self, env):

        if self.noise == 0:
            return self.get_accurate_locations(env) # TODO add a test that at each time step it should give accurate (degrade should happen throuoght time)
        else:
            real_current_location = self.get_accurate_locations(env)

            if self.real_prev_location is None:
                self.real_prev_location = copy.deepcopy(real_current_location)
                self.belief_prev_location = copy.deepcopy(real_current_location)
            else:

                belief_camera_horizon = real_current_location['camera_horizon']
                change_in_xyz = real_current_location['camera_xyz'] - self.real_prev_location['camera_xyz']
                change_in_rotation = real_current_location['camera_rotation'] - self.real_prev_location['camera_rotation']
                belief_camera_xyz = self.add_translation_noise(change_in_xyz, self.belief_prev_location['camera_xyz'])
                belief_camera_rotation = self.add_rotation_noise(change_in_rotation, self.belief_prev_location['camera_rotation'])
                # belief_arm_state = self.add_noise_to_arm(tensor_from_dict(real_current_location['arm_state']['position']), real_agent_location, )
                belief_arm_state = real_current_location['arm_state']

                self.belief_prev_location = copy.deepcopy(dict(camera_xyz=belief_camera_xyz, camera_rotation=belief_camera_rotation, camera_horizon=belief_camera_horizon, arm_state=belief_arm_state))
                self.real_prev_location = copy.deepcopy(real_current_location)


            return self.belief_prev_location
    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        mask = squeeze_bool_mask(self.mask_sensor.get_observation(env, task, *args, **kwargs))
        depth_frame = env.controller.last_event.depth_frame.copy()
        depth_frame[~mask] = -1

        if task.num_steps_taken() == 0:
            self.pointnav_history_aggr = []
            self.real_prev_location = None
            self.belief_prev_location = None


        agent_locations = self.get_agent_localizations(env)

        camera_xyz = agent_locations['camera_xyz']
        camera_rotation = agent_locations['camera_rotation']
        camera_horizon = agent_locations['camera_horizon']
        arm_state = agent_locations['arm_state']

        fov = env.controller.last_event.metadata['fov']


        if mask.sum() != 0:
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
            if self.noise == 0:
                total_sum = [k * v for k,v in self.pointnav_history_aggr]
                total_sum = sum(total_sum)
                total_count = sum([v for k,v in self.pointnav_history_aggr])
                midpoint = total_sum / total_count
                self.pointnav_history_aggr = [(midpoint.cpu(), total_count)]

            else:

                timed_weights = [i + 1 for i in range(len(self.pointnav_history_aggr))]
                total_sum = [timed_weights[i] * self.pointnav_history_aggr[i][0] * self.pointnav_history_aggr[i][1] for i in range(len(self.pointnav_history_aggr))]
                total_count = [v for k,v in self.pointnav_history_aggr]
                real_total_count = [total_count[i] * timed_weights[i] for i in range(len(total_count))]
                midpoint = sum(total_sum) / sum(real_total_count)
                midpoint = midpoint.cpu()

            # agent_centric_middle_of_object = rotate_to_agent(midpoint, self.device, camera_xyz, camera_rotation)
            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            arm_state_agent_coord = convert_world_to_agent_coordinate(arm_state, agent_state)
            distance_in_agent_coord = dict(x=arm_state_agent_coord['position']['x'] - midpoint_agent_coord['position']['x'],y=arm_state_agent_coord['position']['y'] - midpoint_agent_coord['position']['y'],z=arm_state_agent_coord['position']['z'] - midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            # Removing this hurts the performance
            agent_centric_middle_of_object = agent_centric_middle_of_object.abs()
            return agent_centric_middle_of_object

class PointNavEmulatorSensorComplexArm(PointNavEmulatorSensor):

    def average_so_far(self, camera_xyz, camera_rotation, arm_state):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            if self.noise == 0:
                total_sum = [k * v for k,v in self.pointnav_history_aggr]
                total_sum = sum(total_sum)
                total_count = sum([v for k,v in self.pointnav_history_aggr])
                midpoint = total_sum / total_count
                self.pointnav_history_aggr = [(midpoint.cpu(), total_count)]

            else:

                timed_weights = [i + 1 for i in range(len(self.pointnav_history_aggr))]
                total_sum = [timed_weights[i] * self.pointnav_history_aggr[i][0] * self.pointnav_history_aggr[i][1] for i in range(len(self.pointnav_history_aggr))]
                total_count = [v for k,v in self.pointnav_history_aggr]
                real_total_count = [total_count[i] * timed_weights[i] for i in range(len(total_count))]
                midpoint = sum(total_sum) / sum(real_total_count)
                midpoint = midpoint.cpu()

            # agent_centric_middle_of_object = rotate_to_agent(midpoint, self.device, camera_xyz, camera_rotation)

            real_arm_state = self.real_prev_location['arm_state']
            real_camera_xyz = self.real_prev_location['camera_xyz']
            real_camera_rotation = self.real_prev_location['camera_rotation']
            real_agent_state = dict(position=dict(x=real_camera_xyz[0], y=real_camera_xyz[1], z=real_camera_xyz[2], ), rotation=dict(x=0, y=real_camera_rotation, z=0))
            real_arm_state_agent_coord = convert_world_to_agent_coordinate(real_arm_state, real_agent_state)

            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)

            # arm_state_agent_coord = convert_world_to_agent_coordinate(arm_state, agent_state)
            distance_in_agent_coord = dict(x=real_arm_state_agent_coord['position']['x'] - midpoint_agent_coord['position']['x'],y=real_arm_state_agent_coord['position']['y'] - midpoint_agent_coord['position']['y'],z=real_arm_state_agent_coord['position']['z'] - midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            # Removing this hurts the performance
            agent_centric_middle_of_object = agent_centric_middle_of_object.abs()
            return agent_centric_middle_of_object
class PointNavEmulatorSensorOnlyAgentLocation(PointNavEmulatorSensor):

    def average_so_far(self, camera_xyz, camera_rotation, arm_state):
        if len(self.pointnav_history_aggr) == 0:
            return self.dummy_answer
        else:
            if self.noise == 0:
                total_sum = [k * v for k,v in self.pointnav_history_aggr]
                total_sum = sum(total_sum)
                total_count = sum([v for k,v in self.pointnav_history_aggr])
                midpoint = total_sum / total_count
                self.pointnav_history_aggr = [(midpoint.cpu(), total_count)]

            else:

                timed_weights = [i + 1 for i in range(len(self.pointnav_history_aggr))]
                total_sum = [timed_weights[i] * self.pointnav_history_aggr[i][0] * self.pointnav_history_aggr[i][1] for i in range(len(self.pointnav_history_aggr))]
                total_count = [v for k,v in self.pointnav_history_aggr]
                real_total_count = [total_count[i] * timed_weights[i] for i in range(len(total_count))]
                midpoint = sum(total_sum) / sum(real_total_count)
                midpoint = midpoint.cpu()

            # agent_centric_middle_of_object = rotate_to_agent(midpoint, self.device, camera_xyz, camera_rotation)
            agent_state = dict(position=dict(x=camera_xyz[0], y=camera_xyz[1], z=camera_xyz[2], ), rotation=dict(x=0, y=camera_rotation, z=0))
            midpoint_position_rotation = dict(position=dict(x=midpoint[0], y=midpoint[1], z=midpoint[2]), rotation=dict(x=0,y=0,z=0))
            midpoint_agent_coord = convert_world_to_agent_coordinate(midpoint_position_rotation, agent_state)
            distance_in_agent_coord = midpoint_agent_coord['position']

            # arm_state_agent_coord = convert_world_to_agent_coordinate(arm_state, agent_state)
            # distance_in_agent_coord = dict(x=arm_state_agent_coord['position']['x'] - midpoint_agent_coord['position']['x'],y=arm_state_agent_coord['position']['y'] - midpoint_agent_coord['position']['y'],z=arm_state_agent_coord['position']['z'] - midpoint_agent_coord['position']['z'])

            agent_centric_middle_of_object = torch.Tensor([distance_in_agent_coord['x'], distance_in_agent_coord['y'], distance_in_agent_coord['z']])

            # Removing this hurts the performance
            agent_centric_middle_of_object = agent_centric_middle_of_object.abs()


            return agent_centric_middle_of_object

# class PointNavEmulatorSensorwScheduler(PointNavEmulatorSensor):
#     TODO this way of implementing start_noise can be the worst thing that has happened to the humanity and coding
#     def __init__(self,start_noise, type: str, mask_sensor:Sensor,  uuid: str = "point_nav_emul", noise = 0, **kwargs: Any):
#         self.start_noise = start_noise
#         super().__init__(**prepare_locals_for_super(locals()))
#     def get_observation(
#             self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
#     ) -> Any:
#         self.noise = something
#         return super(PointNavEmulatorSensorwScheduler, self).get_observation(env, task, *args, **kwargs)




class PredictionObjectMask(Sensor):
    def __init__(self, type: str,object_query_sensor, rgb_for_detection_sensor,  uuid: str = "predict_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.object_query_sensor = object_query_sensor
        self.rgb_for_detection_sensor = rgb_for_detection_sensor
        uuid = '{}_{}'.format(uuid, type)
        self.device = torch.device("cpu")

        self.detection_model = None

        super().__init__(**prepare_locals_for_super(locals()))

    def load_detection_weights(self):
        self.detection_model = ConditionalDetectionModel()
        detection_weight_dir = '/home/kianae/important_weights/detection_without_color_jitter_model_state_271.pytar'
        if platform.system() == "Darwin":
            detection_weight_dir = '/Users/kianae/important_weights/detection_without_color_jitter_model_state_271.pytar'
        if not os.path.exists(detection_weight_dir):
            detection_weight_dir = detection_weight_dir.replace('/home/kianae', '/home/ubuntu')
        detection_weight_dict = torch.load(detection_weight_dir, map_location='cpu')
        detection_state_dict = self.detection_model.state_dict()
        for key in detection_state_dict:
            param = detection_weight_dict[key]
            detection_state_dict[key].copy_(param)
        remained = [k for k in detection_weight_dict if k not in detection_state_dict]
        # assert len(remained) == 0
        print(
            'WARNING!',
            remained
        )
        self.detection_model.eval()
        self.detection_model.to(self.device) # do i need to assign this

    def get_detection_masks(self, query_images, images):
        query_images = query_images.to(self.device)
        images = images.to(self.device)
        with torch.no_grad():
            batch, c, w, h = images.shape
            predictions = self.detection_model(dict(rgb=images, target_cropped_object=query_images))
            probs_mask = predictions['object_mask']
            mask = probs_mask.argmax(dim=1).float().unsqueeze(1)#To add the channel back in the end of the image
            return mask

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if self.detection_model is None:
            self.load_detection_weights()
        query_object = self.object_query_sensor.get_observation(env, task, *args, **kwargs)
        rgb_frame = self.rgb_for_detection_sensor.get_observation(env, task, *args, **kwargs)
        rgb_frame = torch.Tensor(rgb_frame).permute(2, 0, 1)

        predicted_masks = self.get_detection_masks(query_object.unsqueeze(0), rgb_frame.unsqueeze(0)).squeeze(0)
        # predicted_masks = torch.zeros((1, 224, 224))

        return predicted_masks.permute(1, 2, 0).cpu() #Channel last




class DetectronPredictionObjectMask(Sensor):
    def __init__(self, type: str,object_query_sensor, rgb_for_detection_sensor,  uuid: str = "predict_object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.object_query_sensor = object_query_sensor
        self.rgb_for_detection_sensor = rgb_for_detection_sensor
        uuid = '{}_{}'.format(uuid, type)
        self.device = torch.device("cpu")
        # OMNI_CATEGORIES, ITHOR_TO_OMNI, OMNI_TO_ITHOR
        self.detection_model = None

        super().__init__(**prepare_locals_for_super(locals()))
    def get_detectron_model(self):

        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.model_zoo import model_zoo
        from detectron2.data import MetadataCatalog
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml'))
        # cfg.merge_from_file(model_zoo.get_config_file('LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x'))

        #
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01

        cfg.MODEL.DEVICE = self.device.index if self.device.type != 'cpu' else 'cpu'
        print('loading detectron to ', self.device, cfg.MODEL.DEVICE )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1235
        detection_weight_dir = '/home/kianae/important_weights/detectron2-ithor+lvis-300.pth'
        if platform.system() == "Darwin":
            detection_weight_dir = '/Users/kianae/important_weights/detectron2-ithor+lvis-300.pth'
        if not os.path.exists(detection_weight_dir):
            detection_weight_dir = detection_weight_dir.replace('/home/kianae', '/home/ubuntu')

        # remove
        # detection_weight_dir = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml')
        cfg.MODEL.WEIGHTS = detection_weight_dir
        cfg.INPUT.MIN_SIZE_TEST = 300
        cfg = cfg
        model = DefaultPredictor(cfg)
        # model.eval()
        return model

    def get_detection_masks(self, images, category):

        with torch.no_grad():
            predictions = self.detection_model((images * 255.).astype(np.uint8))
            category_of_interest = OMNI_CATEGORIES.index(ITHOR_TO_OMNI[category])
            boxes = predictions['instances'].pred_boxes
            labels = predictions['instances'].pred_classes
            valid = labels == category_of_interest
            mask = torch.zeros((images.shape[0], images.shape[1]))
            if torch.any(valid):
                valid_boxes = boxes[valid]
                for i in range(len(valid_boxes)):
                    box = valid_boxes[i]
                    x1, y1, x2, y2 = [int(x) for x in box.tensor.squeeze()]
                    mask[y1:y2, x1:x2] = 1
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(224, 224)).squeeze(0).squeeze(0)
            else:
                mask = torch.zeros((224,224))
            return mask.long().cpu().unsqueeze(-1).numpy()#Channel last

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if self.detection_model is None:
            self.detection_model = self.get_detectron_model()
        if self.type == 'source':
            category = task.task_info['source_object_id'].split('|')[0]
        elif self.type == 'destination':
            category = task.task_info['goal_object_id'].split('|')[0]
        # query_object = self.object_query_sensor.get_observation(env, task, *args, **kwargs)
        rgb_frame = self.rgb_for_detection_sensor.get_observation(env, task, *args, **kwargs)
        rgb_frame = rgb_frame[:,:,::-1] #

        predicted_masks = self.get_detection_masks(rgb_frame, category)
        # predicted_masks = torch.zeros((1, 224, 224))

        return predicted_masks#.permute(1, 2, 0).cpu() #Channel last



class RealPointNavSensor(Sensor):

    def __init__(self, type: str, noise=0, uuid: str = "point_nav_real", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.noise = noise
        uuid = '{}_{}'.format(uuid, type)
        self.noise_mode = ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.074, 0.036], [0.019, 0.033]),
                _TruncatedMultivariateGaussian([0.189], [0.038]),
            ),
            rotational_motion=MotionNoiseModel(
                _TruncatedMultivariateGaussian([0.002, 0.003], [0.0, 0.002]),
                _TruncatedMultivariateGaussian([0.219], [0.019]),
            ),
        )


        super().__init__(**prepare_locals_for_super(locals()))

    def get_accurate_locations(self, env):
        metadata = copy.deepcopy(env.controller.last_event.metadata['agent'])
        # camera_xyz = np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]])
        # camera_rotation=metadata["agent"]["rotation"]["y"]
        # camera_horizon=metadata["agent"]["cameraHorizon"]
        # arm_state = env.get_absolute_hand_state()
        return metadata

    def add_translation_noise(self, change_in_xyz, prev_location, real_rotation, belief_rotation):

        if np.abs(change_in_xyz).sum() > 0:
            noise_value_x, noise_value_z = self.noise_mode.linear_motion.linear.sample() * 0.01 * self.noise #to convert to meters
            new_change_in_xyz = change_in_xyz.clone()
            new_change_in_xyz[0] += noise_value_x
            new_change_in_xyz[2] += noise_value_z
            diff_in_rotation = math.radians(belief_rotation - real_rotation)
            # 𝑥2=cos𝛽𝑥1−sin𝛽𝑦1
            # 𝑦2=sin𝛽𝑥1+cos𝛽𝑦1
            new_location = prev_location.clone()
            x = math.cos(diff_in_rotation) * new_change_in_xyz[0] - math.sin(diff_in_rotation) * new_change_in_xyz[2]
            z = math.sin(diff_in_rotation) * new_change_in_xyz[0] + math.cos(diff_in_rotation) * new_change_in_xyz[2]
            new_location[0] += x
            new_location[2] += z
        else:
            new_location = prev_location + change_in_xyz
        return new_location
    def rotate_x_z_around_center(self, x, z, rotation):

        new_x = math.cos(rotation) * x - math.sin(rotation) * z
        new_z = math.sin(rotation) * x + math.cos(rotation) * z

        return new_x, new_z
    def add_rotation_noise(self, change_in_rotation, prev_rotation):
        new_rotation = prev_rotation + change_in_rotation

        if change_in_rotation > 0:
            noise_in_rotation = self.noise_mode.rotational_motion.rotation.sample().item()
            new_rotation += noise_in_rotation
        return new_rotation


    def get_agent_localizations(self, env):

        if self.noise == 0:
            self.real_prev_location = self.get_accurate_locations(env)
            self.belief_prev_location = self.get_accurate_locations(env)
            return self.real_prev_location
        else:
            real_current_location = self.get_accurate_locations(env)

            if self.real_prev_location is None:
                self.real_prev_location = copy.deepcopy(real_current_location)
                self.belief_prev_location = copy.deepcopy(real_current_location)
            else:
                change_in_xyz = tensor_from_dict(real_current_location['position']) - tensor_from_dict(self.real_prev_location['position'])

                last_step_real_rotation, last_step_belief_rotation = self.real_prev_location['rotation']['y'], self.belief_prev_location['rotation']['y']
                change_in_rotation = real_current_location['rotation']['y'] - self.real_prev_location['rotation']['y']
                belief_camera_xyz = self.add_translation_noise(change_in_xyz, tensor_from_dict(self.belief_prev_location['position']), last_step_real_rotation, last_step_belief_rotation)
                belief_camera_rotation = self.add_rotation_noise(change_in_rotation, last_step_belief_rotation)
                # belief_arm_state = self.add_noise_to_arm(tensor_from_dict(real_current_location['arm_state']['position']), real_agent_location, )

                self.belief_prev_location = copy.deepcopy(dict(position=dict(x=belief_camera_xyz[0], y=belief_camera_xyz[1],z=belief_camera_xyz[2]), rotation=dict(x=0,y=belief_camera_rotation, z=0)))
                self.real_prev_location = copy.deepcopy(real_current_location)


            return self.belief_prev_location

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
        if task.num_steps_taken() == 0:
            self.real_prev_location = None
            self.belief_prev_location = None
        goal_obj_id = task.task_info[info_to_search]
        real_object_info = env.get_object_by_id(goal_obj_id)
        real_hand_state = env.get_absolute_hand_state()
        self.get_agent_localizations(env)
        real_agent_state = self.real_prev_location
        belief_agent_state = self.belief_prev_location

        relative_goal_obj = convert_world_to_agent_coordinate(real_object_info, belief_agent_state)
        relative_hand_state = convert_world_to_agent_coordinate(real_hand_state, real_agent_state)

        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = convert_state_to_tensor(dict(position=relative_distance))

        return result


# gimbal lock
class AgentRelativeLocationSensor(Sensor):

    def __init__(self, noise = 0, uuid: str = "agent_relative_location", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.noise = noise
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        agent_initial_state = task.task_info['agent_initial_state']
        current_agent_state = env.controller.last_event.metadata["agent"]

        if self.noise != 0:
            raise Exception('Not implemented yet')
        # To avoid gimbal lock
        def is_close_enough(agent_initial_state, current_agent_state, thr = 0.001):
            initial = [agent_initial_state['position'][k] for k in ['x','y','z']] +[agent_initial_state['rotation'][k] for k in ['x','y','z']]
            current = [current_agent_state['position'][k] for k in ['x','y','z']] +[current_agent_state['rotation'][k] for k in ['x','y','z']]
            for i in range(len(initial)):
                if abs(initial[i] - current[i]) > thr:
                    return False
            return True

        if is_close_enough(agent_initial_state, current_agent_state):
            relative_agent_state = {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}}
        else:
            relative_agent_state = convert_world_to_agent_coordinate(current_agent_state, agent_initial_state)


        # there is something really wrong with convert_world_to_agent_coordinate rotation?

        result = convert_state_to_tensor(relative_agent_state)

        return result


class NoisyDepthSensorThor(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """
    @lazy_property
    def noise_model(self):
        return RedwoodDepthNoise()

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:
        depth = (env.controller.last_event.depth_frame.copy())
        noisy_depth = self.noise_model.add_noise(depth, depth_normalizer=50)
        return noisy_depth

    # @lazy_property
    # def noise_model(self):
    #     from utils.noise_depth_util_files.redwood_depth_noise_model import HabitatRedwoodDepthNoiseModel
    #     return HabitatRedwoodDepthNoiseModel()
    #
    # def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:
    #     depth = (env.controller.last_event.depth_frame.copy())
    #     noisy_depth = self.noise_model.apply(depth / 50) * 50
    #     # cv2.imwrite('/Users/kianae/Desktop/before.png', depth * 25)
    #     # cv2.imwrite('/Users/kianae/Desktop/after.png', noisy_depth * 25)
    #
    #     return noisy_depth


# def not_working_rotate_to_agent(middle_of_object, device, camera_xyz, camera_rotation):
#     recentered_point_cloud = middle_of_object - (torch.FloatTensor([1.0, 0.0, 1.0]).to(device) * camera_xyz).float().reshape((1, 1, 3))
#     # Rotate the cloud so that positive-z is the direction the agent is looking
#     theta = (np.pi * camera_rotation / 180)  # No negative since THOR rotations are already backwards
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     rotation_transform = torch.FloatTensor([[cos_theta, 0, -sin_theta],[0, 1, 0], [sin_theta, 0, cos_theta],]).to(device)
#     rotated_point_cloud = recentered_point_cloud @ rotation_transform.T
#     # xoffset = (map_size_in_cm / 100) / 2
#     # agent_centric_point_cloud = rotated_point_cloud + torch.FloatTensor([xoffset, 0, 0]).to(device)
#     return rotated_point_cloud.squeeze(0).squeeze(0)


class MisDetectionNoisyObjectMask(Sensor):
    def __init__(self, type: str,noise, height, width,  uuid: str = "object_mask", distance_thr: float = -1, misdetection_percent = 1, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.misdetection_percent = misdetection_percent
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
                if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20: # objects that are smaller than this many pixels should be removed. High chance all spatulas will be removed
                    mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        if len(env.controller.last_event.instance_masks) == 0:
            fake_mask = np.zeros(env.controller.last_event.frame[:,:,0].shape)
        else:
            fake_mask = random.choice([v for v in env.controller.last_event.instance_masks.values()])
        fake_mask = (np.expand_dims(fake_mask.astype(np.float),axis=-1))
        # fake_mask, is_real_mask = add_mask_noise(result, fake_mask, noise=self.noise)
        current_shape = fake_mask.shape
        if random.random() < self.misdetection_percent and result.sum() > 0:
            result = fake_mask
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = result
        else:
            resized_mask = cv2.resize(result, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow

        return resized_mask


class MaskCutoffNoisyObjectMask(Sensor):
    def __init__(self, type: str, mask_cutoff_percent,noise, height, width,  uuid: str = "object_mask", distance_thr: float = -1, **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.height = height
        self.width = width
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.mask_cutoff_percent = mask_cutoff_percent
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
                if current_agent_distance_to_obj > self.distance_thr or mask_frame.sum() < 20: # objects that are smaller than this many pixels should be removed. High chance all spatulas will be removed
                    mask_frame[:] = 0

        else:
            mask_frame =np.zeros(env.controller.last_event.frame[:,:,0].shape)

        result = (np.expand_dims(mask_frame.astype(np.float),axis=-1))
        current_shape = result.shape

        if result.sum() > 0:
            w,h,d = current_shape
            mask = np.random.rand(w,h,d)
            mask = mask < self.mask_cutoff_percent
            mask = mask & (result == 1)
            result[mask] = 0
            # plt.imsave('somethingelse.png',result[:,:,0])

        # if random.random() < self.mask_cutoff_percent and result.sum() > 0:
        #     result = fake_mask
        if (current_shape[0], current_shape[1]) == (self.width, self.height):
            resized_mask = result
        else:
            resized_mask = cv2.resize(result, (self.height, self.width)).reshape(self.width, self.height, 1) # my gut says this is gonna be slow

        return resized_mask