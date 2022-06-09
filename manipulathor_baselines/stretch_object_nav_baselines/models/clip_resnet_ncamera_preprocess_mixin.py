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

from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)

from manipulathor_baselines.stretch_object_nav_baselines.models.clip_objnav_ncamera_model import \
    ResnetTensorNavNCameraActorCritic
from manipulathor_utils.debugger_util import ForkedPdb

# from typing import Any, Dict, List, Optional, cast

# import clip
# import gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from allenact.base_abstractions.preprocessor import Preprocessor
# from allenact.utils.misc_utils import prepare_locals_for_super


# class ClipResNetEmbedder(nn.Module):
#     def __init__(self, resnet, pool=True):
#         super().__init__()
#         self.model = resnet
#         self.pool = pool
#         self.eval()

#     def forward(self, x):
#         m = self.model.visual
#         with torch.no_grad():

#             def stem(x):
#                 for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
#                     x = m.relu(bn(conv(x)))
#                 x = m.avgpool(x)
#                 return x

#             x = x.type(m.conv1.weight.dtype)
#             x = stem(x)
#             x = m.layer1(x)
#             x = m.layer2(x)
#             x = m.layer3(x)
#             x = m.layer4(x)
#             if self.pool:
#                 x = F.adaptive_avg_pool2d(x, (1, 1))
#                 x = torch.flatten(x, 1)
#             return x


# class ClipResNetPreprocessor(Preprocessor):
#     """Preprocess RGB or depth image using a ResNet model with CLIP model
#     weights."""

#     CLIP_RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
#     CLIP_RGB_STDS = (0.26862954, 0.26130258, 0.27577711)

#     def __init__(
#         self,
#         rgb_input_uuid: str,
#         clip_model_type: str,
#         pool: bool,
#         device: Optional[torch.device] = None,
#         device_ids: Optional[List[torch.device]] = None,
#         **kwargs: Any,
#     ):
#         assert clip_model_type in clip.available_models()

#         if clip_model_type == "RN50":
#             output_shape = (2048, 7, 7)
#         elif clip_model_type == "RN50x16":
#             output_shape = (3072, 7, 7)
#         else:
#             raise NotImplementedError(
#                 f"Currently `clip_model_type` must be one of 'RN50' or 'RN50x16'"
#             )

#         if pool:
#             output_shape = output_shape[:1]

#         self.clip_model_type = clip_model_type

#         self.pool = pool

#         self.device = torch.device("cpu") if device is None else device
#         self.device_ids = device_ids or cast(
#             List[torch.device], list(range(torch.cuda.device_count()))
#         )
#         self._resnet: Optional[ClipResNetEmbedder] = None

#         low = -np.inf
#         high = np.inf
#         shape = output_shape

#         input_uuids = [rgb_input_uuid]
#         assert (
#             len(input_uuids) == 1
#         ), "resnet preprocessor can only consume one observation type"

#         observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

#         super().__init__(**prepare_locals_for_super(locals()))

#     @property
#     def resnet(self) -> ClipResNetEmbedder:
#         import clip

#         if self._resnet is None:
#             self._resnet = ClipResNetEmbedder(
#                 clip.load(self.clip_model_type, device=self.device)[0], pool=self.pool
#             ).to(self.device)
#         return self._resnet

#     def to(self, device: torch.device) -> "ClipResNetPreprocessor":
#         self._resnet = self.resnet.to(device)
#         self.device = device
#         return self

#     def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
#         x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
#         # If the input is depth, repeat it across all 3 channels
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         x = self.resnet(x).float()
#         return x



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

        # return ResnetTensorNavActorCritic(
        #     action_space=gym.spaces.Discrete(num_actions),
        #     observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
        #     goal_sensor_uuid=goal_sensor_uuid,
        #     rgb_resnet_preprocessor_uuid="rgb_lowres_clip_resnet",
        #     # depth_resnet_preprocessor_uuid="depth_clip_resnet" if has_depth else None,
        #     hidden_size=512,
        #     goal_dims=32,
        #     add_prev_actions=True,
        # )

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