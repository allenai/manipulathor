import platform
from typing import Sequence, Union

from torch import nn
import torch
import yaml

from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel, RGBSensorStretchKinectBigFov
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor


from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment

from manipulathor_baselines.stretch_object_nav_baselines.experiments.obj_nav_base_config import ObjectNavBaseConfig
from utils.stretch_utils.all_rooms_object_nav_task_sampler import AllRoomsObjectNavTaskSampler
from utils.stretch_utils.stretch_object_nav_tasks import StretchObjectNavTask
from manipulathor_utils.debugger_util import ForkedPdb

from manipulathor_baselines.stretch_object_nav_baselines.models.clip_resnet_ncamera_preprocess_mixin import \
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

    WHICH_AGENT = 'stretch'

    # OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))
    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

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

    if platform.system() == "Darwin":
        MAX_STEPS = 100
        # SENSORS += [ #TODO FIX ORDER HERE
        #     RGBSensorStretchKinectBigFov(
        #     height=ObjectNavBaseConfig.SCREEN_SIZE,
        #     width=ObjectNavBaseConfig.SCREEN_SIZE,
        #     use_resnet_normalization=True,
        #     mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
        #     stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        #         uuid="rgb_lowres_arm_only_viz",
        #     ),
        #     RGBSensorThor(
        #     height=ObjectNavBaseConfig.SCREEN_SIZE,
        #     width=ObjectNavBaseConfig.SCREEN_SIZE,
        #     use_resnet_normalization=True,
        #     mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
        #     stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        #         uuid="rgb_lowres_only_viz",
        #     ),
        # ]

    TASK_SAMPLER = AllRoomsObjectNavTaskSampler
    TASK_TYPE = StretchObjectNavTask
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment
    POTENTIAL_VISUALIZERS = [StretchObjNavImageVisualizer, TestMetricLogger]

    TEST_GPU_IDS = list(range(torch.cuda.device_count()))
    NUMBER_OF_TEST_PROCESS = 8

    NUM_PROCESSES = 40
    CLIP_MODEL_TYPE = "RN50"


    def __init__(self):
        super().__init__() 
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #TODO this is nto the best choice
        
        self.preprocessing_and_model = ClipResNetPreprocessNCameraGRUActorCriticMixin(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
        )
        self.REWARD_CONFIG['shaping_weight'] = 1.0
        self.REWARD_CONFIG['exploration_reward'] = 0.05
        self.REWARD_CONFIG['got_stuck_penalty'] = -0.5
        self.REWARD_CONFIG['failed_action_penalty'] = -0.25
        self.ENV_ARGS['renderInstanceSegmentation'] = True

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    
    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=len(self.TASK_TYPE.class_action_names()), **kwargs,
            visualize=self.VISUALIZE
        )


    @classmethod
    def tag(cls):
        return cls.TASK_TYPE.__name__ + '-RGB-2Camera-iTHOR' + '-' +  cls.WHICH_AGENT

    
