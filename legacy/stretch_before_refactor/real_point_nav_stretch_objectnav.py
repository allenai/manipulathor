import gym
from torch import nn

from ithor_arm.ithor_arm_constants import TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    PickedUpObjSensor,
)
from ithor_arm.ithor_arm_viz import TestMetricLogger
from ithor_arm.near_deadline_sensors import RealPointNavSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.stretch_bring_object_baselines.models.stretch_real_pointnav_model import StretchRealPointNavModel
from utils.stretch_utils.stretch_bring_object_task_samplers import StretchDiverseBringObjectTaskSampler
from utils.stretch_utils.stretch_bring_object_tasks import StretchObjectNavTask
from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchIntel, DepthSensorStretchIntel, RGBSensorStretchKinect, DepthSensorStretchKinect, AgentBodyPointNavSensor
from utils.stretch_utils.stretch_visualizer import StretchBringObjImageVisualizer


class RealPointNavStretchObjectNav(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    desired_screen_size = 224
    NOISE_LEVEL = 0
    distance_thr = 1 # is this a good number?
    SENSORS = [
        RGBSensorStretchIntel(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorStretchIntel(
            height=desired_screen_size,
            width=desired_screen_size,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        # RGBSensorStretchKinectZero(
        #     height=desired_screen_size,
        #     width=desired_screen_size,
        #     use_resnet_normalization=True,
        #     uuid="rgb_lowres_arm",
        # ),
        # DepthSensorStretchKinectZero(
        #     height=desired_screen_size,
        #     width=desired_screen_size,
        #     use_normalization=True,
        #     uuid="depth_lowres_arm",
        # ),
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
        PickedUpObjSensor(),
        AgentBodyPointNavSensor(type='source'),
        AgentBodyPointNavSensor(type='destination'),

    ]

    MAX_STEPS = 200

    TASK_SAMPLER = StretchDiverseBringObjectTaskSampler
    TASK_TYPE = StretchObjectNavTask

    NUM_PROCESSES = 40

    OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS

    POTENTIAL_VISUALIZERS = [StretchBringObjImageVisualizer, TestMetricLogger]

    # if platform.system() == "Darwin":
    #     MAX_STEPS = 200

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0#0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 0#1 # is this too big?
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['renderInstanceSegmentation'] = False


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return StretchRealPointNavModel(
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
