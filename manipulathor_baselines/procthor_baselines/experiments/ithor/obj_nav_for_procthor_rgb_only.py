import platform
import random

import gym
import numpy as np
from torch import nn

from ithor_arm.bring_object_sensors import RGBSensorThorNoNan
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from ithor_arm.ithor_arm_sensors import SceneNumberSensor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.procthor_baselines.models.objnav_model_rgb_only import ObjNavOnlyRGBModel
from manipulathor_baselines.procthor_baselines.experiments.ithor.obj_nav_for_procthor import ProcTHORObjectNavBaseConfig
from utils.procthor_utils.procthor_object_nav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import ProcTHORObjectNavTask
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID
from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS


class ProcTHORObjectNavRGBOnly(
    ProcTHORObjectNavBaseConfig
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""
    NOISE_LEVEL = 0
    distance_thr = 1.5 # is this a good number?
    mean = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])
    SENSORS = [
        RGBSensorThorNoNan(
            height=224,
            width=224,
            mean=mean,
            stdev=stdev,
            uuid="rgb_lowres",
        ),
    ]

    MAX_STEPS = 500
    if platform.system() == "Darwin":
        MAX_STEPS = 10

    TASK_SAMPLER = ProcTHORObjectNavTaskSampler
    TASK_TYPE = ProcTHORObjectNavTask
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment

    NUM_PROCESSES = 20

    TRAIN_SCENES = [f'ProcTHOR{i}' for i in range(2)] # 6999
    # if platform.system() == "Darwin":
    #     TRAIN_SCENES = [f'ProcTHOR{i}' for i in range(100)]

    TEST_SCENES = [f'ProcTHOR{i}' for i in range(1)]
    OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list]))


    random.shuffle(TRAIN_SCENES)

    if platform.system() == "Darwin":
        MAX_STEPS = 10


    def __init__(self):
        super().__init__() 
        self.REWARD_CONFIG['goal_success_reward'] = 10.0 
        self.REWARD_CONFIG['step_penalty'] = -0.01 
        self.REWARD_CONFIG['failed_stop_reward'] = 0.0 
        self.REWARD_CONFIG['reached_horizon_reward'] = 0.0
        self.REWARD_CONFIG['shaping_weight'] = 1.0
        
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #TODO this is nto the best choice
        self.ENV_ARGS['scene'] = 'Procedural'
        self.ENV_ARGS['renderInstanceSegmentation'] = 'False'
        self.ENV_ARGS['commit_id'] = PROCTHOR_COMMIT_ID


    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjNavOnlyRGBModel(
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
