from typing import Sequence, Union, Type

import attr
import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.stretch_object_nav_baselines.models.clip_objnav_ncamera_model import \
    ResnetTensorNavNCameraActorCritic
from manipulathor_utils.debugger_util import ForkedPdb


@attr.s(kw_only=True)
class ClipResNetPreprocessNCameraGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    clip_model_type: str = attr.ib()
    screen_size: int = attr.ib()
    pool: bool = attr.ib(default=False)

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        self.resnet_preprocessor_uuids = []
        for camera in [s for s in self.sensors if (isinstance(s, RGBSensor) or isinstance(s,DepthSensor))]:
            if "_only_viz" not in camera.uuid:
                if isinstance(camera, RGBSensor):
                    assert (
                        np.linalg.norm(
                            np.array(camera._norm_means)
                            - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
                        )
                        < 1e-5
                    )
                    assert (
                        np.linalg.norm(
                            np.array(camera._norm_sds)
                            - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
                        )
                        < 1e-5
                    )
                preprocessors.append(
                    ClipResNetPreprocessor(
                        rgb_input_uuid=camera.uuid,
                        clip_model_type=self.clip_model_type,
                        pool=self.pool,
                        output_uuid=camera.uuid+"_clip_resnet",
                    )
                )
                self.resnet_preprocessor_uuids.append(camera.uuid+"_clip_resnet")

            else:
                self.resnet_preprocessor_uuids.append(camera.uuid)

        return preprocessors

    def create_model(self, num_actions: int, visualize: bool, **kwargs) -> nn.Module:
        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )
        self.resnet_preprocessor_uuids = [s for s in self.resnet_preprocessor_uuids if "_only_viz" not in s]

        # display or assert sensor order here? possible source for sneaky failure if they're not the same 
        # as in pretraining when loading weights.

        return ResnetTensorNavNCameraActorCritic(
            action_space=gym.spaces.Discrete(num_actions),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            resnet_preprocessor_uuids=self.resnet_preprocessor_uuids,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=True,
            visualize=visualize
        )