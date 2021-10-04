import gym
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.stretch_bring_object_baselines.stretch_utils.stretch_bring_object_task_samplers import StretchDiverseBringObjectTaskSampler
from manipulathor_baselines.stretch_bring_object_baselines.stretch_utils.stretch_bring_object_tasks import ExploreWiseRewardTaskObjNav
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS


class ComplexRewardObjectNavStretch(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 #TODO is this a good number?
    desired_screen_size = 224
    SENSORS = [
        RGBSensorThor(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=desired_screen_size,
            width=desired_screen_size,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        PickedUpObjSensor(),
        CategorySampleSensor(type='source'),
        CategorySampleSensor(type='destination'),
        NoisyObjectMask(noise=NOISE_LEVEL, type='source', distance_thr=distance_thr, height=desired_screen_size, width=desired_screen_size),
        NoisyObjectMask(noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr, height=desired_screen_size, width=desired_screen_size),
    ]

    #TODO we should add done to this experiment

    MAX_STEPS = 200

    TASK_SAMPLER = StretchDiverseBringObjectTaskSampler
    TASK_TYPE = ExploreWiseRewardTaskObjNav

    NUM_PROCESSES = 40

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS



    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 #TODO is this too big?
        self.REWARD_CONFIG['object_found'] = 1 #TODO is this too big?

        # assert ( This is not true anymore, resize them appropriately
        #         self.CAMERA_WIDTH == 224
        #         and self.CAMERA_HEIGHT == 224
        #         and self.VISIBILITY_DISTANCE == 1
        #         and self.STEP_SIZE == 0.25
        # )

        #TODO camera height and width is going to be different
        self.ENV_ARGS = {**STRETCH_ENV_ARGS, "renderDepthImage": True}


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
