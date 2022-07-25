import platform
import yaml

from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_wide_stochastic \
    import ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVNoisy
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor


class ProcTHORObjectNavClipResnet50RGBOnly1CameraNarrowFOV(
    ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVNoisy
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)
        

    SENSORS = [
        RGBSensorStretchIntel(
            height=ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVNoisy.SCREEN_SIZE,
            width=ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVNoisy.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=OBJECT_TYPES,
        ),
    ]

    if platform.system() == "Darwin":
        SENSORS += [
            RGBSensorStretchIntel(
            height=ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVNoisy.SCREEN_SIZE,
            width=ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOVNoisy.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_only_viz",
            ),
        ]
    
    @classmethod
    def tag(cls):
        return cls.TASK_TYPE.__name__ + '-RGB-1Camera-ProcTHOR-narrowFOV' + '-' +  cls.WHICH_AGENT