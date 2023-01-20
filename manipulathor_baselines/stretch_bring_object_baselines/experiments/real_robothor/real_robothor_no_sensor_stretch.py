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
from manipulathor_baselines.stretch_bring_object_baselines.models.stretch_no_pointnav_model import \
    StretchNoPointNavModel
from manipulathor_baselines.stretch_bring_object_baselines.models.stretch_pointnav_emul_model import StretchPointNavEmulModel
from manipulathor_baselines.stretch_bring_object_baselines.models.stretch_real_pointnav_model import StretchRealPointNavModel
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS, KITCHEN_TRAIN, LIVING_ROOM_TRAIN, \
    BEDROOM_TRAIN, ROBOTHOR_TRAIN, ROBOTHOR_VAL, BATHROOM_TEST, BATHROOM_TRAIN, BEDROOM_TEST, LIVING_ROOM_TEST, \
    KITCHEN_TEST
from utils.stretch_utils.real_stretch_bring_object_task_sampler import RealStretchDiverseBringObjectTaskSampler
# from utils.stretch_utils.real_stretch_sensors import RealRGBSensorStretchNav, RealDepthSensorStretchIntel, \
#     RealRGBSensorStretchManip, RealDepthSensorStretchManip, RealStretchPickedUpObjSensor, StretchDetectronObjectMask
from utils.stretch_utils.real_stretch_sensors import RealRGBSensorStretchNav, RealRGBSensorStretchManip, \
    StretchDetectronObjectMask, RealDepthSensorStretchNav, RealDepthSensorStretchManip, RealStretchPickedUpObjSensor
from utils.stretch_utils.real_stretch_tasks import RealStretchExploreWiseRewardTask
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, STRETCH_MANIPULATHOR_COMMIT_ID, INTEL_CAMERA_WIDTH


class RealPointNavEmulStretchAllRooms(
    StretchBringObjectiThorBaseConfig,
    StretchBringObjectMixInPPOConfig,
    StretchBringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    desired_screen_size = INTEL_CAMERA_WIDTH
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?

    rgb_intel_camera_sensor = RealRGBSensorStretchNav(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres_raw",
        )
    rgb_kinect_camera_sensor = RealRGBSensorStretchManip(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres_arm_raw",
        )


    source_mask_sensor_intel = StretchDetectronObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='source', distance_thr=distance_thr, only_close_big_masks=True, source_camera=rgb_intel_camera_sensor, uuid='object_mask')
    destination_mask_sensor_intel = StretchDetectronObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='destination', distance_thr=distance_thr, only_close_big_masks=True, source_camera=rgb_intel_camera_sensor, uuid='object_mask')

    source_mask_sensor_kinect = StretchDetectronObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='source', distance_thr=distance_thr, only_close_big_masks=True, source_camera=rgb_kinect_camera_sensor, uuid='object_mask_kinect')
    destination_mask_sensor_kinect = StretchDetectronObjectMask(height=desired_screen_size, width=desired_screen_size,noise=0, type='destination', distance_thr=distance_thr, only_close_big_masks=True, source_camera=rgb_kinect_camera_sensor, uuid='object_mask_kinect')

    SENSORS = [
        RealRGBSensorStretchNav(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        RealDepthSensorStretchNav(height=desired_screen_size,width=desired_screen_size,use_normalization=True,uuid="depth_lowres",),
        RealRGBSensorStretchManip(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres_arm",
        ),
        RealDepthSensorStretchManip(
            height=desired_screen_size,
            width=desired_screen_size,
            use_normalization=True,
            uuid="depth_lowres_arm",
        ),
        RealStretchPickedUpObjSensor(),
        source_mask_sensor_intel,
        destination_mask_sensor_intel,
        source_mask_sensor_kinect,
        destination_mask_sensor_kinect,

    ]

    MAX_STEPS = 200

    TASK_SAMPLER = RealStretchDiverseBringObjectTaskSampler
    TASK_TYPE = RealStretchExploreWiseRewardTask

    NUM_PROCESSES = 1
    NUMBER_OF_TEST_PROCESS = 1



    TRAIN_SCENES = ['RealRobothor']
    TEST_SCENES = ['RealRobothor']
    OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list if room_typ == 'robothor']))


    random.shuffle(TRAIN_SCENES)

    VISUALIZE = True



    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1
        self.REWARD_CONFIG['object_found'] = 1
        # self.ENV_ARGS = STRETCH_ENV_ARGS
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['commit_id'] = STRETCH_MANIPULATHOR_COMMIT_ID
        self.ENV_ARGS['renderInstanceSegmentation'] = True

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return StretchNoPointNavModel(
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
