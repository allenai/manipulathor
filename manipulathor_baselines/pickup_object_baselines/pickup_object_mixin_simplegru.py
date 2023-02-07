from typing import Sequence, Union

import gym
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
# from allenact.base_abstractions.sensor import RGBSensor, DepthSensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, Sensor, RGBSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.bring_object_baselines.experiments.bring_object_base import BringObjectBaseConfig
from manipulathor_baselines.bring_object_baselines.models.bring_object_models import BringObjectBaselineActorCritic
from manipulathor_baselines.pickup_object_baselines.pickup_object_base import PickUpObjectBaseConfig


class PickUpObjectMixInSimpleGRUConfig(PickUpObjectBaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:

        return PickUpObjectBaselineActorCritic(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )
