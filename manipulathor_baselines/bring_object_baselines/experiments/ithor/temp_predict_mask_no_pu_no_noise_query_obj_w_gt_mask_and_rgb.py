import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, TempAllMasksSensor, TempEpisodeNumber, TempObjectCategorySensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pickup_object_with_mask_model import PickUpWMaskBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.predict_mask_small_bring_object_model import SmallBringObjectWPredictMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_small_bring_object_model import SmallBringObjectWQueryObjGtMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.rgbd_w_predict_mask_small_bring_object_model import PredictMaskSmallBringObjectWQueryObjRGBDModel
from manipulathor_baselines.bring_object_baselines.models.small_bring_object_with_mask_model import SmallBringObjectWMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.small_depth_pickup_object_with_mask_model import SmallPickUpWMaskDepthBaselineActorCritic


class PredictMaskNoNoiseRGBQueryObjGTMaskSimpleDiverseBringObject(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    # NOISE_LEVEL = 0
    SENSORS = [
        NoGripperRGBSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="only_detection_rgb_lowres",
        ),
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
        NoisyObjectMask(noise=0.0, type='source', uuid='gt_mask_for_loss'),
        NoisyObjectMask(noise=0.0, type='destination', uuid='gt_mask_for_loss'),
        TempAllMasksSensor(),
        TempEpisodeNumber(),
        TempObjectCategorySensor(type='source'), #TODO take these source and destination to the sensor itself
        TempObjectCategorySensor(type='destination'),
    ]

    MAX_STEPS = 200
    TASK_SAMPLER = DiverseBringObjectTaskSampler
    NUM_PROCESSES = 40

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PredictMaskSmallBringObjectWQueryObjRGBDModel(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
            visualize=cls.VISUALIZE,
        )

    @classmethod
    def tag(cls):
        return cls.__name__
