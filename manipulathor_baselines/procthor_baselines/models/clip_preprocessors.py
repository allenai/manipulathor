from typing import Any, Dict, List, Optional, cast
from typing_extensions import Literal

import clip
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super
from manipulathor_utils.debugger_util import ForkedPdb
from utils.hacky_viz_utils import hacky_visualization
from datetime import datetime


class ClipResNetEmbedder(nn.Module):
    def __init__(self, resnet, pool=True):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.eval()

    def forward(self, x):
        m = self.model.visual
        with torch.no_grad():

            def stem(x):
                for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                    x = m.relu(bn(conv(x)))
                x = m.avgpool(x)
                return x

            x = x.type(m.conv1.weight.dtype)
            x = stem(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            if self.pool:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
            return x

class ClipResNetPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model with CLIP model
    weights."""

    def __init__(
        self,
        rgb_input_uuid: str,
        clip_model_type: Literal["RN50", "RN50x16"],
        pool: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        visualize=False, # TODO this is such a hack job
        **kwargs: Any,
    ):
        assert clip_model_type in clip.available_models()

        if clip_model_type == "RN50":
            output_shape = (2048, 7, 7)
        elif clip_model_type == "RN50x16":
            output_shape = (3072, 7, 7)
        else:
            raise NotImplementedError(
                f"Currently `clip_model_type` must be one of 'RN50' or 'RN50x16'"
            )

        if pool:
            output_shape = output_shape[:1]

        self.clip_model_type = clip_model_type

        self.pool = pool

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self._resnet: Optional[ClipResNetEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = output_shape

        input_uuids = [rgb_input_uuid]
        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)
        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))
        self.visualize = visualize

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def resnet(self) -> ClipResNetEmbedder:
        import clip

        if self._resnet is None:
            self._resnet = ClipResNetEmbedder(
                clip.load(self.clip_model_type, device=self.device)[0], pool=self.pool
            ).to(self.device)
        return self._resnet

    def to(self, device: torch.device) -> "ClipResNetPreprocessor":
        self._resnet = self.resnet.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # if self.visualize:
        #     obs['rgb_lowres'] = x.permute(0,2,3,1).unsqueeze(0)
        #     hacky_visualization(obs, base_directory_to_right_images=self.starting_time)
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x).float()
        return x


class ClipTextPreprocessor(Preprocessor):
    def __init__(
        self,
        goal_sensor_uuid: str,
        object_types: List[str],
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any,
    ):
        try:
            import clip

            self.clip = clip
        except ImportError as _:
            raise ImportError(
                "Cannot `import clip` when instatiating `CLIPResNetPreprocessor`."
                " Please install clip from the openai/CLIP git repository:"
                "\n`pip install git+https://github.com/openai/CLIP.git@3b473b0e682c091a9e53623eebc1ca1657385717`"
            )

        output_shape = (1024,)

        self.object_types = object_types

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        low = -np.inf
        high = np.inf
        shape = output_shape

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        input_uuids = [goal_sensor_uuid]

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def text_encoder(self):
        if self._clip_model is None:
            self._clip_model = self.clip.load("RN50", device=self.device)[0]
            self._clip_model.eval()
        return self._clip_model.encode_text

    def to(self, device: torch.device):
        self.device = device
        self._clip_model = None
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        object_inds = obs[self.input_uuids[0]]
        object_types = [self.object_types[ind] for ind in object_inds]
        x = self.clip.tokenize([f"navigate to the {obj}" for obj in object_types]).to(
            self.device
        )
        with torch.no_grad():
            return self.text_encoder(x).float()
