import platform
import random

import gym
import torch
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, \
    CategoryFeatureSampleSensor, DepthSensorThorNoNan
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask, ExploreWiseRewardTask
from ithor_arm.ithor_arm_constants import MANIPULATHOR_ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from ithor_arm.near_deadline_sensors import PointNavEmulatorSensor, RealPointNavSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
from manipulathor_baselines.procthor_baselines.models.objdis_pointnav_model import ObjDisPointNavModel
from manipulathor_baselines.procthor_baselines.models.objdis_pointnav_only_rgb_model import ObjDisPointNavOnlyRGBModel
from manipulathor_baselines.procthor_baselines.models.pnemul_model_simple import PointNavEmulModelSimple
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import KITCHEN_TRAIN, BEDROOM_TRAIN, BATHROOM_TRAIN, \
    BATHROOM_TEST, BEDROOM_TEST, LIVING_ROOM_TEST, KITCHEN_TEST, LIVING_ROOM_TRAIN, FULL_LIST_OF_OBJECTS
from utils.procthor_utils.all_rooms_obj_dis_task_sampler import AllRoomsBringObjectTaskSampler
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID, STRETCH_MANIPULATHOR_COMMIT_ID


class PNEmulObjDisArmPointNavITHORAllRooms(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?
    source_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='source', distance_thr=distance_thr)
    destination_mask_sensor = NoisyObjectMask(height=BringObjectiThorBaseConfig.SCREEN_SIZE, width=BringObjectiThorBaseConfig.SCREEN_SIZE,noise=NOISE_LEVEL, type='destination', distance_thr=distance_thr)
    # no_normalization_depth = DepthSensorThorNoNan( TODO be sure that the no nan depth is reflecting everywhere
    #         height=BringObjectiThorBaseConfig.SCREEN_SIZE,
    #         width=BringObjectiThorBaseConfig.SCREEN_SIZE,
    #         use_normalization=False,
    #         uuid="depth_lowres_nonorm",
    #     )
    SENSORS = [
        RGBSensorThor(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThorNoNan(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        PickedUpObjSensor(),
        source_mask_sensor,
        destination_mask_sensor,
        PointNavEmulatorSensor(type='source', mask_sensor=source_mask_sensor, depth_sensor=no_normalization_depth),
        PointNavEmulatorSensor(type='destination', mask_sensor=destination_mask_sensor, depth_sensor=no_normalization_depth),
        # RealPointNavSensor(type='source', uuid='arm_point_nav'),
        # RealPointNavSensor(type='destination', uuid='arm_point_nav'),
        # TempRealArmpointNav(uuid='point_nav_emul',type='source'),
        # TempRealArmpointNav(uuid='point_nav_emul', type='destination'),
    ]

    MAX_STEPS = 200

    TASK_SAMPLER = AllRoomsBringObjectTaskSampler
    # TASK_TYPE = TestPointNavExploreWiseRewardTask
    TASK_TYPE = ExploreWiseRewardTask
    ENVIRONMENT_TYPE = ManipulaTHOREnvironment

    NUM_PROCESSES = 20

    TRAIN_SCENES = KITCHEN_TRAIN + LIVING_ROOM_TRAIN + BEDROOM_TRAIN + BATHROOM_TRAIN
    TEST_SCENES = KITCHEN_TEST + LIVING_ROOM_TEST + BEDROOM_TEST + BATHROOM_TEST
    OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list if room_typ != 'robothor']))

    random.shuffle(TRAIN_SCENES)
    random.shuffle(TEST_SCENES)

    # if platform.system() == "Darwin":
    #     MAX_STEPS = 10

    def test_task_sampler_args(self, **kwargs):
        sampler_args = super(PNEmulObjDisArmPointNavITHORAllRooms, self).test_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:
            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, PointNavEmulatorSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])
        return sampler_args

    def train_task_sampler_args(self, **kwargs):
        sampler_args = super(PNEmulObjDisArmPointNavITHORAllRooms, self).train_task_sampler_args(**kwargs)
        if platform.system() == "Darwin":
            pass
        else:

            for sensor_type in sampler_args['sensors']:
                if isinstance(sensor_type, PointNavEmulatorSensor):
                    sensor_type.device = torch.device(kwargs["devices"][0])

        return sampler_args

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #  this is nto the best choice
        self.ENV_ARGS['commit_id'] = xxx#STRETCH_MANIPULATHOR_COMMIT_ID TODO test with same commit id?
        self.ENV_ARGS['renderInstanceSegmentation'] = True

        self.ENV_ARGS['renderDepthImage'] = True


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavEmulModelSimple(
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
