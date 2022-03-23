import platform
import random

import gym
import torch
from torch import nn


from manipulathor_baselines.stretch_bring_object_baselines.experiments.stretch_bring_object_ithor_base import \
    StretchBringObjectiThorBaseConfig
from manipulathor_baselines.stretch_bring_object_baselines.experiments.stretch_bring_object_mixin_ddppo import \
    StretchBringObjectMixInPPOConfig
from manipulathor_baselines.stretch_bring_object_baselines.experiments.stretch_bring_object_mixin_simplegru import \
    StretchBringObjectMixInSimpleGRUConfig
from manipulathor_baselines.stretch_bring_object_baselines.models.stretch_pointnav_emul_model import StretchPointNavEmulModel
from manipulathor_baselines.stretch_bring_object_baselines.models.stretch_real_pointnav_model import StretchRealPointNavModel
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS, KITCHEN_TRAIN, LIVING_ROOM_TRAIN, \
    BEDROOM_TRAIN, ROBOTHOR_TRAIN, ROBOTHOR_VAL, BATHROOM_TEST, BATHROOM_TRAIN, BEDROOM_TEST, LIVING_ROOM_TEST, \
    KITCHEN_TEST
from utils.stretch_utils.stretch_bring_object_task_samplers import StretchDiverseBringObjectTaskSampler
from utils.stretch_utils.stretch_bring_object_tasks import StretchExploreWiseRewardTask, \
    StretchExploreWiseRewardTaskOnlyPickUp, StretchObjectNavTask
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, STRETCH_MANIPULATHOR_COMMIT_ID, INTEL_CAMERA_WIDTH
from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchIntel, DepthSensorStretchIntel, \
    RGBSensorStretchKinect, DepthSensorStretchKinect, AgentBodyPointNavSensor, AgentBodyPointNavEmulSensor, \
    RGBSensorStretchKinectZero, \
    DepthSensorStretchKinectZero, IntelRawDepthSensor, ArmPointNavEmulSensor, KinectRawDepthSensor, \
    KinectNoisyObjectMask, IntelNoisyObjectMask, StretchPickedUpObjSensor
from utils.stretch_utils.stretch_visualizer import StretchBringObjImageVisualizer


class PointNavEmulStretchAllRooms(
    StretchBringObjectiThorBaseConfig,
    StretchBringObjectMixInPPOConfig,
    StretchBringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    desired_screen_size = INTEL_CAMERA_WIDTH
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?

    source_mask_sensor_intel = IntelNoisyObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='source', distance_thr=distance_thr, only_close_big_masks=True)
    destination_mask_sensor_intel = IntelNoisyObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='destination', distance_thr=distance_thr, only_close_big_masks=True)
    depth_sensor_intel = IntelRawDepthSensor()

    source_mask_sensor_kinect = KinectNoisyObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='source', distance_thr=distance_thr, only_close_big_masks=True)
    destination_mask_sensor_kinect = KinectNoisyObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='destination', distance_thr=distance_thr, only_close_big_masks=True)
    depth_sensor_kinect = KinectRawDepthSensor()


    SENSORS = [
        RGBSensorStretchIntel(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorStretchIntel(height=desired_screen_size,width=desired_screen_size,use_normalization=True,uuid="depth_lowres",),
        RGBSensorStretchKinect(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres_arm",
        ),
        DepthSensorStretchKinect(
            height=desired_screen_size,
            width=desired_screen_size,
            use_normalization=True,
            uuid="depth_lowres_arm",
        ),
        StretchPickedUpObjSensor(),
        AgentBodyPointNavEmulSensor(type='source', mask_sensor=source_mask_sensor_intel, depth_sensor=depth_sensor_intel),
        AgentBodyPointNavEmulSensor(type='destination', mask_sensor=destination_mask_sensor_intel, depth_sensor=depth_sensor_intel),
        ArmPointNavEmulSensor(type='source', mask_sensor=source_mask_sensor_kinect, depth_sensor=depth_sensor_kinect),
        ArmPointNavEmulSensor(type='destination', mask_sensor=destination_mask_sensor_kinect, depth_sensor=depth_sensor_kinect),
        source_mask_sensor_intel,
        destination_mask_sensor_intel,
        source_mask_sensor_kinect,
        destination_mask_sensor_kinect,

    ]

    MAX_STEPS = 200

    TASK_SAMPLER = StretchDiverseBringObjectTaskSampler
    TASK_TYPE = StretchExploreWiseRewardTask

    NUM_PROCESSES = 30

    TRAIN_SCENES = ROBOTHOR_TRAIN
    TEST_SCENES = ROBOTHOR_VAL
    OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list if room_typ == 'robothor']))


    random.shuffle(TRAIN_SCENES)





    # if platform.system() == "Darwin": TODO remove
    #     random.shuffle(KITCHEN_TRAIN)
    #     random.shuffle(LIVING_ROOM_TRAIN)
    #     random.shuffle(BEDROOM_TRAIN)
    #     random.shuffle(BATHROOM_TRAIN)
    #     random.shuffle(ROBOTHOR_TRAIN)
    #     TRAIN_SCENES = KITCHEN_TRAIN[:10] + LIVING_ROOM_TRAIN[:10]  + BEDROOM_TRAIN[:10]  + BATHROOM_TRAIN[:10]  + ROBOTHOR_TRAIN[:10]

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1
        self.REWARD_CONFIG['object_found'] = 1
        # self.ENV_ARGS = STRETCH_ENV_ARGS
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['commit_id'] = STRETCH_MANIPULATHOR_COMMIT_ID
        self.ENV_ARGS['renderInstanceSegmentation'] = True

    def test_task_sampler_args(self, **kwargs):
        sampler_args = super(PointNavEmulStretchAllRooms, self).test_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:
            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, AgentBodyPointNavEmulSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])

            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, ArmPointNavEmulSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])

        return sampler_args

    def train_task_sampler_args(self, **kwargs):
        sampler_args = super(PointNavEmulStretchAllRooms, self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, AgentBodyPointNavEmulSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])
            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, ArmPointNavEmulSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])
        return sampler_args


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return StretchPointNavEmulModel(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
            visualize=cls.VISUALIZE
        )

    @classmethod
    def tag(cls):
        return cls.__name__
