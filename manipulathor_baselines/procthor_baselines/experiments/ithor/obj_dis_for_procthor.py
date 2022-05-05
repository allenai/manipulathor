import platform
import random

import gym
import torch
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, \
    CategoryFeatureSampleSensor, RGBSensorThorNoNan, DepthSensorThorNoNan
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
from scripts.dataset_generation.find_categories_to_use import KITCHEN_TRAIN, BEDROOM_TRAIN, BATHROOM_TRAIN, \
    BATHROOM_TEST, BEDROOM_TEST, LIVING_ROOM_TEST, KITCHEN_TEST, LIVING_ROOM_TRAIN, FULL_LIST_OF_OBJECTS
from utils.procthor_utils.all_rooms_obj_dis_task_sampler import AllRoomsBringObjectTaskSampler
from utils.procthor_utils.procthor_bring_object_task_samplers import ProcTHORDiverseBringObjectTaskSampler
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID


class ObjDisArmPointNavProcTHOR(
    BringObjectiThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?
    SENSORS = [
        RGBSensorThorNoNan(
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
        RealPointNavSensor(type='source', uuid='arm_point_nav'),
        RealPointNavSensor(type='destination', uuid='arm_point_nav'),
        # TempRealArmpointNav(uuid='point_nav_emul',type='source'),
        # TempRealArmpointNav(uuid='point_nav_emul', type='destination'),
    ]

    MAX_STEPS = 200

    TASK_SAMPLER = ProcTHORDiverseBringObjectTaskSampler
    # TASK_TYPE = TestPointNavExploreWiseRewardTask
    TASK_TYPE = ExploreWiseRewardTask
    ENVIRONMENT_TYPE = ManipulaTHOREnvironment

    NUM_PROCESSES = 20

    TRAIN_SCENES = [f'ProcTHOR{i}' for i in range(6999)]
    if platform.system() == "Darwin":
        TRAIN_SCENES = [f'ProcTHOR{i}' for i in range(100)]

    TEST_SCENES = KITCHEN_TEST + LIVING_ROOM_TEST + BEDROOM_TEST + BATHROOM_TEST
    OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list]))


    random.shuffle(TRAIN_SCENES)

    if platform.system() == "Darwin":
        MAX_STEPS = 10

    # remove
    #
    # TEST_SCENES = [f'FloorPlan{i + 1}_physics' for i in range(5)]
    # OBJECT_TYPES = OBJECT_TYPES[:3]
    # TEST_SCENES = [f'FloorPlan{i + 1}_physics' for i in range(1)]
    # OBJECT_TYPES = ['Egg', 'Spatula']
    # MAX_STEPS = 10




    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #TODO this is nto the best choice
        self.ENV_ARGS['commit_id'] = PROCTHOR_COMMIT_ID


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjDisPointNavModel(
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
