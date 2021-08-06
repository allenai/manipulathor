import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import TargetObjectBBox, TargetLocationBBox, CategorySampleSensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer, ImageVisualizer
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pickup_object_with_mask_model import PickUpWMaskBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.predict_mask_small_bring_object import SmallBringObjectWPredictMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_small_bring_object import SmallBringObjectWQueryObjGtMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.small_bring_object_with_mask_model import SmallBringObjectWMaskDepthBaselineActorCritic
from manipulathor_baselines.bring_object_baselines.models.small_depth_pickup_object_with_mask_model import SmallPickUpWMaskDepthBaselineActorCritic
from utils.locobot_utils.locobot_bring_object_task_sampler import LocoBotDiverseBringObjectTaskSampler
from utils.locobot_utils.locobot_sensors import LocoBotPickedUpObjSensor, LocoBotCategorySampleSensor, LocoBotObjectMask


class LocoBotGTMaskSimpleDiverseBringObject(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="only_detection_rgb_lowres",
        ),
        DepthSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        LocoBotPickedUpObjSensor(),
        LocoBotCategorySampleSensor(type='source'),
        LocoBotCategorySampleSensor(type='destination'),
        LocoBotObjectMask(noise=0, type='source'),
        LocoBotObjectMask(noise=0, type='destination'),
    ]

    MAX_STEPS = 20000
    POTENTIAL_VISUALIZERS = [ImageVisualizer]

    # POTENTIAL_VISUALIZERS = BringObjectiThorBaseConfig.POTENTIAL_VISUALIZERS + [MaskImageVisualizer]



    TASK_SAMPLER = LocoBotDiverseBringObjectTaskSampler
    NUM_PROCESSES = 1

    TRAIN_SCENES = ['FloorPlan1_physics']
    TEST_SCENES = ['FloorPlan1_physics']
    TRAIN_OBJECTS = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug"]
    TEST_OBJECTS = ["Pan", "Egg", "Spatula", "Cup"] #, 'Potato']
    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS



    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return SmallBringObjectWQueryObjGtMaskDepthBaselineActorCritic(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )

    @classmethod
    def tag(cls):
        return cls.__name__
