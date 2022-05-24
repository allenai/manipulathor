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

from manipulathor_baselines.procthor_baselines.experiments.ithor.obj_nav_for_procthor import ProcTHORObjectNavBaseConfig
from utils.procthor_utils.procthor_object_nav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import StretchObjectNavTask, ObjectNavTask
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID, STRETCH_ENV_ARGS
from manipulathor_utils.debugger_util import ForkedPdb

from manipulathor_baselines.procthor_baselines.models.clip_preprocessors import ClipResNetPreprocessor
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic
from manipulathor_baselines.procthor_baselines.models.clip_objnav_ncamera_model import ResnetTensorNavNCameraActorCritic
from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from ithor_arm.ithor_arm_viz import TestMetricLogger



class ProcTHORObjectNavClipResnet50RGBOnly(
    ProcTHORObjectNavBaseConfig
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    NOISE_LEVEL = 0
    distance_thr = 1.0 # match procthor config
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    stdev = np.array([0.26862954, 0.26130258, 0.27577711])
    SENSORS = [
        RGBSensorThor(
            height=224,
            width=224,
            use_resnet_normalization=True,
            mean=mean,
            stdev=stdev,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=OBJECT_TYPES,
        ),
    ]



    TASK_SAMPLER = ProcTHORObjectNavTaskSampler
    TASK_TYPE = StretchObjectNavTask
    ENVIRONMENT_TYPE = StretchManipulaTHOREnvironment
    POTENTIAL_VISUALIZERS = [StretchObjNavImageVisualizer, TestMetricLogger]


    NUM_PROCESSES = 40
    # NUM_TRAIN_HOUSES = 40 # set None or comment out for all
    MAX_STEPS = 500
    if platform.system() == "Darwin":
        MAX_STEPS = 10
        NUM_TRAIN_HOUSES = 40
    VISUALIZE = False

    def __init__(self):
        super().__init__() 

        self.ENV_ARGS = copy.deepcopy(STRETCH_ENV_ARGS)
        self.ENV_ARGS['p_randomize_material'] = 0.8
        self.ENV_ARGS['visibilityDistance'] = self.distance_thr
        self.ENV_ARGS['environment_type'] = self.ENVIRONMENT_TYPE #TODO this is nto the best choice
        self.ENV_ARGS['scene'] = 'Procedural'
        self.ENV_ARGS['renderInstanceSegmentation'] = False
        self.ENV_ARGS['renderDepthImage'] = False
        self.ENV_ARGS['commit_id'] = PROCTHOR_COMMIT_ID
        self.ENV_ARGS['allow_flipping'] = False


    @classmethod
    @final
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        rgb_sensor = next((s for s in cls.SENSORS if isinstance(s, RGBSensorThor)), None)

        if rgb_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=rgb_sensor.uuid,
                    clip_model_type="RN50",
                    pool=False,
                    output_uuid="rgb_clip_resnet",
                    visualize=cls.VISUALIZE
                )
            )

        return preprocessors

    # @classmethod
    # @final
    # def create_model(cls, **kwargs) -> nn.Module:
    #     goal_sensor_uuid = next(
    #         (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
    #         None,
    #     )
    #     resnet_preprocessor_uuids = ["rgb_clip_resnet"]

    #     return ResnetTensorNavNCameraActorCritic(
    #         action_space=gym.spaces.Discrete(len(cls.TASK_TYPE.class_action_names())),
    #         observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
    #         goal_sensor_uuid=goal_sensor_uuid,
    #         resnet_preprocessor_uuids=resnet_preprocessor_uuids,
    #         hidden_size=512,
    #         goal_dims=32,
    #         add_prev_actions=True,
    #     )
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensorThor) for s in cls.SENSORS)
        has_depth = False #any(isinstance(s, DepthSensor) for s in cls.SENSORS)

        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_clip_resnet" if has_depth else None,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=True #cfg.model.add_prev_actions_embedding,
        )

    @classmethod
    def tag(cls):
        return cls.__name__    
