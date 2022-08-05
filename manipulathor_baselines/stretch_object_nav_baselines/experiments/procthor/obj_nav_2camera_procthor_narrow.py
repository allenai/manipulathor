import platform
import yaml

from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_wide \
    import ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor

from manipulathor_baselines.stretch_object_nav_baselines.models.clip_resnet_ncamera_preprocess_mixin \
    import TaskIdSensor

class ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV(
    ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    # NUM_PROCESSES = 1
    # NUMBER_OF_TEST_PROCESS = 0
    # NUMBER_OF_VALID_PROCESS = 0
        

    SENSORS = [
        RGBSensorStretchIntel(
            height=2*ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            width=2*ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        RGBSensorStretchKinect(
            height=2*ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            width=2*ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres_arm",
        ),
        GoalObjectTypeThorSensor(
            object_types=OBJECT_TYPES,
        ),
        TaskIdSensor(),
    ]

    @classmethod
    def tag(cls):
        return cls.TASK_TYPE.__name__ + '-RGB-2Camera-ProcTHOR-narrowFOV' + '-' +  cls.WHICH_AGENT
