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
from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel, RGBSensorStretchKinectBigFov
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor


from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment

from manipulathor_baselines.object_nav_baselines.experiments.obj_nav_base_config import ObjectNavBaseConfig
from utils.procthor_utils.all_rooms_object_nav_task_sampler import AllRoomsObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import StretchObjectNavTask
from utils.stretch_utils.stretch_constants import STRETCH_MANIPULATHOR_COMMIT_ID, STRETCH_ENV_ARGS
from manipulathor_utils.debugger_util import ForkedPdb

# from manipulathor_baselines.procthor_baselines.models.clip_preprocessors import ClipResNetPreprocessor
from manipulathor_baselines.object_nav_baselines.models.clip_resnet_ncamera_preprocess_mixin import \
    ClipResNetPreprocessNCameraGRUActorCriticMixin
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder
from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from ithor_arm.ithor_arm_viz import TestMetricLogger



class ithorObjectNavClipResnet50RGBOnly2CameraWideFOV(
    ObjectNavBaseConfig
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    TRAIN_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, 20 + 1)
    ]
    VALID_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(21, 26)
    ]
    TEST_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(26, 31)
    ]

    ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VALID_SCENES

    # OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))
    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    NOISE_LEVEL = 0
    distance_thr = 1.0 # match procthor config
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    stdev = np.array([0.26862954, 0.26130258, 0.27577711])
    SENSORS = [
        RGBSensorThor(
            height=ObjectNavBaseConfig.SCREEN_SIZE,
            width=ObjectNavBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        RGBSensorStretchKinectBigFov(
            height=ObjectNavBaseConfig.SCREEN_SIZE,
            width=ObjectNavBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres_arm",
        ),
        GoalObjectTypeThorSensor(
            object_types=OBJECT_TYPES,
        ),
    ]

    MAX_STEPS = 500
    if platform.system() == "Darwin":
        MAX_STEPS = 100
        SENSORS += [
            RGBSensorStretchKinectBigFov(
            height=ObjectNavBaseConfig.SCREEN_SIZE,
            width=ObjectNavBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_arm_only_viz",
            ),
            RGBSensorThor(
            height=ObjectNavBaseConfig.SCREEN_SIZE,
            width=ObjectNavBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_only_viz",
            ),
        ]

    TASK_SAMPLER = AllRoomsObjectNavTaskSampler
    TASK_TYPE = StretchObjectNavTask
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment
    POTENTIAL_VISUALIZERS = [StretchObjNavImageVisualizer, TestMetricLogger]

    NUM_PROCESSES = 40
    CLIP_MODEL_TYPE = "RN50"


    def __init__(self):
        super().__init__() 

        self.ENV_ARGS = copy.deepcopy(STRETCH_ENV_ARGS)
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #TODO this is nto the best choice
        self.ENV_ARGS['renderInstanceSegmentation'] = False
        self.ENV_ARGS['commit_id'] = STRETCH_MANIPULATHOR_COMMIT_ID
        
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


    @classmethod
    def tag(cls):
        return cls.__name__

    
