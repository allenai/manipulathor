import platform
import yaml

from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.procthor_baselines.experiments.ithor.obj_nav_for_procthor_clip_resnet50_rgb_only import ProcTHORObjectNavClipResnet50RGBOnly
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor


class ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV(
    ProcTHORObjectNavClipResnet50RGBOnly
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    SENSORS = [
        RGBSensorStretchIntel(
            height=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            width=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        RGBSensorStretchKinect(
            height=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            width=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
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
        SENSORS += [
            RGBSensorStretchKinect(
            height=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            width=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_arm_only_viz",
            ),
            RGBSensorStretchIntel(
            height=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            width=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_only_viz",
            ),
        ]
