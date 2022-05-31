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

from manipulathor_baselines.procthor_baselines.experiments.ithor.obj_nav_2camera_procthor_wide import \
    ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV
from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor


from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment

from manipulathor_baselines.procthor_baselines.experiments.ithor.obj_nav_for_procthor import ProcTHORObjectNavBaseConfig
from utils.procthor_utils.procthor_object_nav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import StretchObjectNavTask
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID, STRETCH_ENV_ARGS
from manipulathor_utils.debugger_util import ForkedPdb

from manipulathor_baselines.procthor_baselines.models.clip_preprocessors import ClipResNetPreprocessor
from manipulathor_baselines.procthor_baselines.models.clip_objnav_ncamera_model import ResnetTensorNavNCameraActorCritic
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder
from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from ithor_arm.ithor_arm_viz import TestMetricLogger
from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorNavActorCritic



class ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV(
    ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV
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
    if platform.system() == "Darwin":
        MAX_STEPS = 10
        SENSORS += [
            RGBSensorStretchKinect(
                height=224,
                width=224,
                use_resnet_normalization=True,
                mean=mean,
                stdev=stdev,
                uuid="rgb_lowres_arm_only_viz",
            ),
            RGBSensorStretchIntel(
                height=224,
                width=224,
                use_resnet_normalization=True,
                mean=mean,
                stdev=stdev,
                uuid="rgb_lowres_only_viz",
            ),
        ]

