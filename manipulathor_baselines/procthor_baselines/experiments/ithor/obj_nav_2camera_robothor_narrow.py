import platform
import random
from typing import Sequence, Union
from typing_extensions import final

import gym
import numpy as np
from torch import nn
import yaml
import copy

from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor


from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment

from manipulathor_baselines.procthor_baselines.experiments.ithor.obj_nav_base_config import ObjectNavBaseConfig
from utils.procthor_utils.all_rooms_object_nav_task_sampler import AllRoomsObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import StretchObjectNavTask
from utils.stretch_utils.stretch_constants import STRETCH_MANIPULATHOR_COMMIT_ID, STRETCH_ENV_ARGS
from manipulathor_utils.debugger_util import ForkedPdb

from manipulathor_baselines.procthor_baselines.models.clip_preprocessors import ClipResNetPreprocessor
from manipulathor_baselines.procthor_baselines.models.clip_objnav_ncamera_model import ResnetTensorNavNCameraActorCritic
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder
from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from ithor_arm.ithor_arm_viz import TestMetricLogger

from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS, ROBOTHOR_TRAIN, ROBOTHOR_VAL



class RobothorObjectNavClipResnet50RGBOnly2CameraNarrowFOV(
    ObjectNavBaseConfig
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    TRAIN_SCENES = ROBOTHOR_TRAIN
    TEST_SCENES = ROBOTHOR_VAL
    # OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list if room_typ == 'robothor']))
    # OBJECT_TYPES.sort()

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)


    random.shuffle(TRAIN_SCENES)
    random.shuffle(TEST_SCENES)

    NOISE_LEVEL = 0
    distance_thr = 1.0 # match procthor config
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    stdev = np.array([0.26862954, 0.26130258, 0.27577711])
    SENSORS = [
        RGBSensorStretchIntel(
            height=224,
            width=224,
            use_resnet_normalization=True,
            mean=mean,
            stdev=stdev,
            uuid="rgb_lowres",
        ),
        RGBSensorStretchKinect(
            height=224,
            width=224,
            use_resnet_normalization=True,
            mean=mean,
            stdev=stdev,
            uuid="rgb_lowres_arm",
        ),
        GoalObjectTypeThorSensor(
            object_types=OBJECT_TYPES,
        ),
    ]

    MAX_STEPS = 500
    if platform.system() == "Darwin":
        MAX_STEPS = 500

    TASK_SAMPLER = AllRoomsObjectNavTaskSampler
    TASK_TYPE = StretchObjectNavTask
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment
    POTENTIAL_VISUALIZERS = [StretchObjNavImageVisualizer, TestMetricLogger]

    NUM_PROCESSES = 40


    def __init__(self):
        super().__init__() 

        self.ENV_ARGS = copy.deepcopy(STRETCH_ENV_ARGS)
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #TODO this is nto the best choice
        self.ENV_ARGS['renderInstanceSegmentation'] = False
        self.ENV_ARGS['commit_id'] = STRETCH_MANIPULATHOR_COMMIT_ID


    @classmethod
    @final
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        # rgb_sensor = next((s for s in cls.SENSORS if isinstance(s, RGBSensorThor)), None)

        preprocessors.append(
            ClipResNetPreprocessor(
                rgb_input_uuid="rgb_lowres",
                clip_model_type="RN50",
                pool=False,
                output_uuid="rgb_clip_resnet",
                visualize=cls.VISUALIZE
            )
        )
        preprocessors.append(
            ClipResNetPreprocessor(
                rgb_input_uuid="rgb_lowres_arm",
                clip_model_type="RN50",
                pool=False,
                output_uuid="rgb_clip_resnet_arm",
                visualize=cls.VISUALIZE
            )
        )
        return preprocessors

    @classmethod
    @final
    def create_model(cls, **kwargs) -> nn.Module:
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )
        resnet_preprocessor_uuids = ["rgb_clip_resnet","rgb_clip_resnet_arm"]

        return ResnetTensorNavNCameraActorCritic(
            action_space=gym.spaces.Discrete(len(cls.TASK_TYPE.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            resnet_preprocessor_uuids=resnet_preprocessor_uuids,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=True,
        )

    @classmethod
    def tag(cls):
        return cls.__name__

    
