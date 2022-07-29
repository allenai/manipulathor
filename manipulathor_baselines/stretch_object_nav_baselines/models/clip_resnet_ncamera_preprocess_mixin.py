from typing import Sequence, Union, Type, Any

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

from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)

from manipulathor_baselines.stretch_object_nav_baselines.models.clip_objnav_ncamera_model import \
    ResnetTensorNavNCameraActorCritic
from manipulathor_utils.debugger_util import ForkedPdb

class TaskIdSensor(Sensor):
    def __init__(
        self,
        uuid: str = "task_id_sensor",
        **kwargs: Any,
    ):
        super().__init__(uuid=uuid, observation_space=gym.spaces.Discrete(1))

    def _get_observation_space(self):
        if self.target_to_detector_map is None:
            return gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            return gym.spaces.Discrete(len(self.detector_types))

    def get_observation(
        self,
        env,
        task,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if task.task_info["id"] is not None:
            out = [ord(k) for k in task.task_info["id"]]
            for _ in range(len(out), 1000):
                out.append(ord(" "))
            return out
        else:
            return 1


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