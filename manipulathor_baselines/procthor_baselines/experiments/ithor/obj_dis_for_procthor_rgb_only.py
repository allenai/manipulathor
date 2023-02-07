import platform
import random

import gym
import numpy as np
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
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, SceneNumberSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from ithor_arm.near_deadline_sensors import PointNavEmulatorSensor, RealPointNavSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.pointnav_emulator_model import RGBDModelWPointNavEmulator
from manipulathor_baselines.procthor_baselines.experiments.procthor_base_config import BringObjectProcThorBaseConfig
from manipulathor_baselines.procthor_baselines.models.objdis_pointnav_model import ObjDisPointNavModel
from manipulathor_baselines.procthor_baselines.models.objdis_pointnav_only_rgb_model import ObjDisPointNavOnlyRGBModel
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import KITCHEN_TRAIN, BEDROOM_TRAIN, BATHROOM_TRAIN, \
    BATHROOM_TEST, BEDROOM_TEST, LIVING_ROOM_TEST, KITCHEN_TEST, LIVING_ROOM_TRAIN, FULL_LIST_OF_OBJECTS
from utils.procthor_utils.all_rooms_obj_dis_task_sampler import AllRoomsBringObjectTaskSampler
from utils.procthor_utils.procthor_bring_object_task_samplers import ProcTHORDiverseBringObjectTaskSampler
from utils.procthor_utils.procthor_helper import PROCTHOR_INVALID_SCENES
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID


class ObjDisArmPointNavRGBOnlyProcTHOR(
    BringObjectProcThorBaseConfig,
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
        PickedUpObjSensor(),
        # SceneNumberSensor(), TODO remove as soon as bug is resolved
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


    TEST_SCENES = BringObjectProcThorBaseConfig.TRAIN_SCENES

    # if platform.system() == "Darwin":
    #     MAX_STEPS = 10

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
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE
        self.ENV_ARGS['scene'] = 'Procedural'
        self.ENV_ARGS['renderInstanceSegmentation'] = False
        self.ENV_ARGS['renderDepthImage'] = False
        self.ENV_ARGS['commit_id'] = PROCTHOR_COMMIT_ID



    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjDisPointNavOnlyRGBModel(
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