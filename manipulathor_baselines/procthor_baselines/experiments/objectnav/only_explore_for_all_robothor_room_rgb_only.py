import copy
import platform
import random
from typing import Sequence

import gym
import torch
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)
from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor, CategoryFeatureSampleSensor
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
from manipulathor_baselines.procthor_baselines.experiments.procthor_base_config import BringObjectProcThorBaseConfig
from manipulathor_baselines.procthor_baselines.models.clip_resnet_objectnav import \
    ClipResNetPreprocessGRUActorCriticMixinObjectNav
from manipulathor_baselines.procthor_baselines.models.clip_resnet_preprocess_mixin import \
    ClipResNetPreprocessGRUActorCriticMixin
from manipulathor_baselines.procthor_baselines.models.objdis_pointnav_model import ObjDisPointNavModel
from manipulathor_baselines.procthor_baselines.models.objdis_pointnav_only_rgb_model import ObjDisPointNavOnlyRGBModel
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import KITCHEN_TRAIN, BEDROOM_TRAIN, BATHROOM_TRAIN, \
    BATHROOM_TEST, BEDROOM_TEST, LIVING_ROOM_TEST, KITCHEN_TEST, LIVING_ROOM_TRAIN, FULL_LIST_OF_OBJECTS, \
    ROBOTHOR_TRAIN, ROBOTHOR_VAL
from utils.procthor_utils.all_rooms_obj_dis_task_sampler import AllRoomsBringObjectTaskSampler
from utils.procthor_utils.procthor_bring_object_task_samplers import ProcTHORDiverseBringObjectTaskSampler
from utils.procthor_utils.procthor_objectnav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.procthor_utils.robothor_objectnav_task_sampler import RoboTHORObjectNavTaskSampler
from utils.stretch_utils.stretch_bring_object_tasks import StretchObjectNavTask, StretchObjectNavTaskwLookUp, \
    StretchOnlyExploreTaskwLookUp
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID, STRETCH_MANIPULATHOR_COMMIT_ID, STRETCH_ENV_ARGS, \
    MIN_HORIZON, MAX_HORIZON
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment, \
    StretchManipulaTHOREnvironmentwNoisyFailedActions
from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchIntel, RGBSensorStretchKinect, \
    RGBSensorStretchIntelwJitter, RGBSensorStretchKinectwJitter


class OnlyExploreAllRoboTHORRoomsRGBOnly(
    BringObjectProcThorBaseConfig,
    BringObjectMixInPPOConfig,
    BringObjectMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?

    LIST_OF_OBJECT_TYPES = ['Apple', 'BaseballBat', 'BasketBall', 'Bed', 'Bottle', 'Chair', 'CoffeeTable', 'FloorLamp', 'HousePlant', 'Laptop', 'Sofa', 'Television'] #TODO this needs to be changed later
    OBJECT_TO_SEARCH = ['Apple']

    SENSORS = [
        # RGBSensorStretchIntelwJitter(
        RGBSensorStretchIntel(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        # RGBSensorStretchIntelwJitter(
        RGBSensorStretchIntel(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="only_detection_rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        # RGBSensorStretchKinectwJitter(
        RGBSensorStretchKinect(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres_arm",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        # RGBSensorStretchKinectwJitter(
        RGBSensorStretchKinect(
            height=BringObjectiThorBaseConfig.SCREEN_SIZE,
            width=BringObjectiThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="only_detection_rgb_lowres_arm",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),

        GoalObjectTypeThorSensor(
            object_types=LIST_OF_OBJECT_TYPES,
        ),
    ]

    MAX_STEPS = 200 #TODO do we want to increase this?

    if platform.system() == "Darwin":
        MAX_STEPS = 10

    TASK_SAMPLER = RoboTHORObjectNavTaskSampler
    # TASK_TYPE = TestPointNavExploreWiseRewardTask
    # TASK_TYPE = StretchObjectNavTask
    TASK_TYPE = StretchOnlyExploreTaskwLookUp
    # ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironmentwNoisyFailedActions
    OBJECT_TYPES = OBJECT_TO_SEARCH


    REAL_ROOM_COMMIT_ID = '47b45bbfd14e9cef767b3ff7d9056ddd52f69ab8'

    NUM_PROCESSES = 20 #
    TRAIN_SCENES = ROBOTHOR_TRAIN
    TEST_SCENES = ROBOTHOR_VAL


    CLIP_MODEL_TYPE = "RN50"
    FIRST_CAMERA_HORIZON = [i for i in range(MIN_HORIZON, MAX_HORIZON, 10)] #

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG['exploration_reward'] = 0.1 # is this too big?
        self.REWARD_CONFIG['object_found'] = 1 # is this too big?
        self.ENV_ARGS = copy.deepcopy(STRETCH_ENV_ARGS)
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE
        # self.ENV_ARGS['scene'] = 'Procedural'
        self.ENV_ARGS['renderInstanceSegmentation'] = False
        self.ENV_ARGS['renderDepthImage'] = False
        # self.ENV_ARGS['commit_id'] = PROCTHOR_COMMIT_ID
        self.ENV_ARGS['commit_id'] = self.REAL_ROOM_COMMIT_ID

        self.preprocessing_and_model = ClipResNetPreprocessGRUActorCriticMixinObjectNav(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
        )

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=len(self.TASK_TYPE.class_action_names()), **kwargs,
            visualize=self.VISUALIZE
        )
    def preprocessors(self):
        return self.preprocessing_and_model.preprocessors()



    @classmethod
    def tag(cls):
        return cls.__name__
