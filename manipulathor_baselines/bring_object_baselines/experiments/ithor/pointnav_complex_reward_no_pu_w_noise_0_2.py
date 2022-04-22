import platform

import gym
import torch
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, CategoryFeatureSampleSensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask, ExploreWiseRewardTask
from ithor_arm.ithor_arm_constants import MANIPULATHOR_ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from ithor_arm.near_deadline_sensors import PointNavEmulatorSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator


class PointNavNewModelAndHandWAgentNoise(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    AGENT_LOCATION_NOISE = 0.2 #?
    distance_thr = 1.5 # is this a good number?
    source_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr)
    destination_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr)
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
        CategorySampleSensor(type='source'),
        CategorySampleSensor(type='destination'),
        CategoryFeatureSampleSensor(type='source'),
        CategoryFeatureSampleSensor(type='destination'),
        source_mask_sensor,
        destination_mask_sensor,
        PointNavEmulatorSensor(type='source', mask_sensor=source_mask_sensor, noise=AGENT_LOCATION_NOISE),
        PointNavEmulatorSensor(type='destination', mask_sensor=destination_mask_sensor, noise=AGENT_LOCATION_NOISE),
        # TempRealArmpointNav(uuid='point_nav_emul',type='source'),
        # TempRealArmpointNav(uuid='point_nav_emul', type='destination'),
    ]

    MAX_STEPS = 200

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    # TASK_TYPE = TestPointNavExploreWiseRewardTask
    TASK_TYPE = ExploreWiseRewardTask

    NUM_PROCESSES = 20

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS


    def train_task_sampler_args(self, **kwargs):
        sampler_args = super(PointNavNewModelAndHandWAgentNoise, self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for pointnav_emul_sensor in sampler_args['sensors']:
                if isinstance(pointnav_emul_sensor, PointNavEmulatorSensor):
                    pointnav_emul_sensor.device = torch.device(kwargs["devices"][0])

        return sampler_args
    def test_task_sampler_args(self, **kwargs):
        sampler_args = super(PointNavNewModelAndHandWAgentNoise, self).test_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for pointnav_emul_sensor in sampler_args['sensors']:
                if isinstance(pointnav_emul_sensor, PointNavEmulatorSensor):
                    pointnav_emul_sensor.device = torch.device(kwargs["devices"][0])

        return sampler_args

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return RGBDModelWPointNavEmulator(
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
