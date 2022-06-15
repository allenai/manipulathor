import platform
import yaml

from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchKinectBigFov
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_for_procthor_clip_resnet50_rgb_only \
    import ProcTHORObjectNavClipResnet50RGBOnly
from utils.stretch_utils.stretch_object_nav_tasks import StretchObjectNavTask
from manipulathor_utils.debugger_util import ForkedPdb

from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor



class ProcTHORObjectNavClipResnet50RGBOnly2CameraWideFOV(
    ProcTHORObjectNavClipResnet50RGBOnly
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    
    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    SENSORS = [
        RGBSensorThor(
            height=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            width=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),

        RGBSensorStretchKinectBigFov(
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

    # NUM_PROCESSES = 40
    # NUM_TRAIN_HOUSES = None

    if platform.system() == "Darwin":
        SENSORS += [
            RGBSensorStretchKinectBigFov(
                height=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
                width=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
                use_resnet_normalization=True,
                mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
                stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_arm_only_viz",
            ),
            RGBSensorThor(
                height=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
                width=ProcTHORObjectNavClipResnet50RGBOnly.SCREEN_SIZE,
                use_resnet_normalization=True,
                mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
                stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_only_viz",
            ),
        ]

    def __init__(self):
        super().__init__()
        assert (
                self.WHICH_AGENT == 'stretch' # this only works for stretch
                and self.ENV_ARGS['allow_flipping'] == False # not with 2-camera
        )

    @classmethod
    def tag(cls):
        return cls.TASK_TYPE.__name__ + '-RGB-2Camera-ProcTHOR' + '-' +  cls.WHICH_AGENT

    
