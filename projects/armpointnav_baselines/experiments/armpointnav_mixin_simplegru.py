from typing import Sequence, Union

import gym
import torch.nn as nn
from torchvision import models

from core.base_abstractions.preprocessor import Preprocessor
from core.base_abstractions.preprocessor import ResNetPreprocessor
from core.base_abstractions.sensor import RGBSensor, DepthSensor
from plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from plugins.ithor_arm_plugin.ithor_arm_tasks import AbstractPickUpDropOffTask
from projects.armpointnav_baselines.experiments.armpointnav_base import ArmPointNavBaseConfig
from projects.armpointnav_baselines.models.arm_pointnav_models import (
    ArmPointNavBaselineActorCritic,
)
from utils.experiment_utils import Builder


class ArmPointNavMixInSimpleGRUConfig(ArmPointNavBaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ArmPointNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )

