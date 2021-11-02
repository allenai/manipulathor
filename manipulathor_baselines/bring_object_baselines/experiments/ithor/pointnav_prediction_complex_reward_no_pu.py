import platform
import time

import gym
import torch
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, CategoryFeatureSampleSensor, TempAllMasksSensor, TempEpisodeNumber, TempObjectCategorySensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask, TestPointNavExploreWiseRewardTask, ExploreWiseRewardTask
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from ithor_arm.near_deadline_sensors import PointNavEmulatorSensor, PredictionObjectMask
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.detection_model import ConditionalDetectionModel
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
from manipulathor_baselines.bring_object_baselines.models.predict_super_simple_pointnav_emulator_model import PredictSuperSimpleRGBDModelWPointNavEmulator
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.temp_super_simple_pointnav_emulator_model import SuperSimpleRGBDModelWPointNavEmulator


class PredictionPointNavNoPUNewModelAndHand(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    #TODO do we want to add binary head later?
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?

    source_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr)
    destination_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr)
    source_object_category_sensor = TempObjectCategorySensor(type='source')
    destination_object_category_sensor = TempObjectCategorySensor(type='destination')
    category_sample_source = CategorySampleSensor(type='source')
    category_sample_destination = CategorySampleSensor(type='destination')
    rgb_for_detection_sensor = NoGripperRGBSensorThor(
        height=BringObjectiThorBaseConfig.SCREEN_SIZE,
        width=BringObjectiThorBaseConfig.SCREEN_SIZE,
        use_resnet_normalization=True,
        uuid="only_detection_rgb_lowres",
    )
    source_mask_sensor_prediction = PredictionObjectMask(type='source',object_query_sensor=category_sample_source, rgb_for_detection_sensor=rgb_for_detection_sensor)
    destination_mask_sensor_prediction = PredictionObjectMask(type='destination',object_query_sensor=category_sample_destination, rgb_for_detection_sensor=rgb_for_detection_sensor)


    SENSORS = [
        RGBSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        PickedUpObjSensor(),
        CategoryFeatureSampleSensor(type='source'),
        CategoryFeatureSampleSensor(type='destination'),
        category_sample_source,
        category_sample_destination,
        source_mask_sensor,
        destination_mask_sensor,
        source_mask_sensor_prediction,
        destination_mask_sensor_prediction,
        PointNavEmulatorSensor(type='source', mask_sensor=source_mask_sensor_prediction),
        PointNavEmulatorSensor(type='destination', mask_sensor=destination_mask_sensor_prediction),
    ]

    MAX_STEPS = 200

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    # TASK_TYPE = TestPointNavExploreWiseRewardTask
    TASK_TYPE = ExploreWiseRewardTask

    NUM_PROCESSES = 20
    NUMBER_OF_TEST_PROCESS = 5 #TODO is there anyway to faster?
    # TEST_GPU_IDS = list(range(torch.cuda.device_count()))
    # print('TEST_GPU_IDS', TEST_GPU_IDS)


    # VISUALIZE = True

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS


    def test_task_sampler_args(self, **kwargs):
        sampler_args = super(type(self), self).test_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:
            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, PointNavEmulatorSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])
                if isinstance(sensor_type, PredictionObjectMask):
                    sensor_type.device = torch.device(kwargs["devices"][0])
        return sampler_args

    def train_task_sampler_args(self, **kwargs): #TODO you have to specify it in the call to train_task_sampler_args (or valid/test_task_sampler_args). For now maybe you can just add something like:
        sampler_args = super(type(self), self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, PointNavEmulatorSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])
                if isinstance(sensor_type, PredictionObjectMask):
                    sensor_type.device = torch.device(kwargs["devices"][0])

        return sampler_args

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PredictSuperSimpleRGBDModelWPointNavEmulator(
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
