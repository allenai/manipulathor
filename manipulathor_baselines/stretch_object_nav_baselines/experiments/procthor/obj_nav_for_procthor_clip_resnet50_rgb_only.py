import platform
import random
from typing import Sequence, Union
from typing_extensions import final

import gym
import numpy as np
from torch import nn
import yaml
import copy

from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor


from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from utils.stretch_utils.stretch_ithor_arm_environment_minimal import MinimalStretchManipulaTHOREnvironment

from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_for_procthor import ProcTHORObjectNavBaseConfig
from utils.procthor_utils.procthor_object_nav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.stretch_utils.stretch_object_nav_tasks import StretchObjectNavTask, ObjectNavTask, StretchNeckedObjectNavTask
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS
from manipulathor_utils.debugger_util import ForkedPdb

from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from manipulathor_baselines.stretch_object_nav_baselines.models.clip_resnet_ncamera_preprocess_mixin import \
    ClipResNetPreprocessNCameraGRUActorCriticMixin
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder


class ProcTHORObjectNavClipResnet50RGBOnly(
    ProcTHORObjectNavBaseConfig
):
    """Single-camera Object Navigation experiment configuration in ProcTHOR, using CLIP preprocessing."""

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    NOISE_LEVEL = 0
    distance_thr = 1.0 # set larger for stretch work
    WHICH_AGENT = 'locobot' # 'locobot' 'default' 'stretch'

    SENSORS = [
        RGBSensorThor(
            height=ProcTHORObjectNavBaseConfig.SCREEN_SIZE,
            width=ProcTHORObjectNavBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=OBJECT_TYPES,
        ),
    ]

    MAX_STEPS = 500
    if platform.system() == "Darwin":
        MAX_STEPS = 200
        NUM_TRAIN_HOUSES = 100
        SENSORS += [
            RGBSensorThor(
                height=ProcTHORObjectNavBaseConfig.SCREEN_SIZE,
                width=ProcTHORObjectNavBaseConfig.SCREEN_SIZE,
                use_resnet_normalization=True,
                mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
                stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_only_viz",
            ),
        ]

    NUM_PROCESSES = 56

    TASK_SAMPLER = ProcTHORObjectNavTaskSampler
    TASK_TYPE = StretchNeckedObjectNavTask
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment

    # NUM_TRAIN_HOUSES = 500

    CLIP_MODEL_TYPE = "RN50"


    def __init__(self):
        super().__init__() 

        # self.ENV_ARGS = copy.deepcopy(STRETCH_ENV_ARGS)
        # self.ENV_ARGS['agentMode']=self.WHICH_AGENT
        self.ENV_ARGS['p_randomize_material'] = 0.8
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #TODO this is nto the best choice
        self.ENV_ARGS['renderInstanceSegmentation'] = False
        self.ENV_ARGS['renderDepthImage'] = False        
        self.ENV_ARGS['allow_flipping'] = True
        # if self.WHICH_AGENT == 'locobot':
        #     self.ENV_ARGS['fieldOfView'] = 63.453048374758716
        #     self.ENV_ARGS['rotateStepDegrees'] = 30.0
        #     self.ENV_ARGS["snapToGrid"] = False
            # self.ENV_ARGS["quality"] = 'Ultra'
            # self.ENV_ARGS['width'] = 400
            # self.ENV_ARGS['height'] = 300
        
        # self.REWARD_CONFIG['shaping_weight'] = 0.0

        self.preprocessing_and_model = ClipResNetPreprocessNCameraGRUActorCriticMixin(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    
    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=len(self.TASK_TYPE.class_action_names()), **kwargs,
            visualize=self.VISUALIZE
        )

    def get_agent(self):
        return self.ENV_ARGS['agentMode']

    @classmethod
    def tag(cls):
        return cls.TASK_TYPE.__name__ + '-RGB-SingleCam-ProcTHOR' + '-' +  cls.WHICH_AGENT
