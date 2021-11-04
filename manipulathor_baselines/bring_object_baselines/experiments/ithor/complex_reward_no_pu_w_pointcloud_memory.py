import platform

import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, PointCloudMemory, CategoryFeatureSampleSensor, NoMaskSensor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask, ExploreWiseRewardTask
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
from manipulathor_baselines.bring_object_baselines.models.gt_mask_with_memory_model import MemoryWGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel



class ComplexRewardNoPUWPointCloudMemory(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SCREEN_SIZE = 224
    NOISE_LEVEL = 0 #TODO need to put this back 0.2
    distance_thr = 100 # is this a good number?
    source_object_mask = NoisyObjectMask(height=224, width=224,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr)
    destination_object_mask = NoisyObjectMask(height=224, width=224,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr)

    SENSORS = [
        RGBSensorThor(
            height=224,
            width=224,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=224,
            width=224,
            use_normalization=True,
            uuid="depth_lowres",
        ),

        CategoryFeatureSampleSensor(type='source'),
        CategoryFeatureSampleSensor(type='destination'),

        PickedUpObjSensor(),
        CategorySampleSensor(type='source'),
        CategorySampleSensor(type='destination'),

        source_object_mask,
        destination_object_mask,
        PointCloudMemory(uuid='point_cloud', memory_size=5, mask_generator=source_object_mask),
    ]

    MAX_STEPS = 200

    TASK_SAMPLER = DiverseBringObjectTaskSampler
    TASK_TYPE = ExploreWiseRewardTask


    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS

    TEST_SCENES = ['FloorPlan1_physics']
    OBJECT_TYPES = ['Pot', 'Pan']
    # OBJECT_TYPES = ['Lettuce', 'Apple']



    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?



    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return SmallBringObjectWQueryObjGtMaskRGBDModel(
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
