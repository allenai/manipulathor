from utils.stretch_utils.stretch_constants import INTEL_CAMERA_WIDTH
from manipulathor_utils.debugger_util import ForkedPdb

from utils.stretch_utils.real_stretch_sensors import RealRGBSensorStretchIntel, RealRGBSensorStretchKinect
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor

from utils.stretch_utils.real_stretch_environment import StretchRealEnvironment
from utils.stretch_utils.all_rooms_object_nav_task_sampler import RealStretchAllRoomsObjectNavTaskSampler
from utils.stretch_utils.stretch_object_nav_tasks import RealStretchObjectNavTask


from manipulathor_baselines.stretch_object_nav_baselines.experiments.ithor.obj_nav_2camera_ithor_wide import \
     ithorObjectNavClipResnet50RGBOnly2CameraWideFOV


class RealStretchObjectNav(
    ithorObjectNavClipResnet50RGBOnly2CameraWideFOV
):
    desired_screen_size = INTEL_CAMERA_WIDTH
    SENSORS = [
        RealRGBSensorStretchIntel(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        RealRGBSensorStretchKinect(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres_arm",
        ),
        GoalObjectTypeThorSensor(
            object_types=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.OBJECT_TYPES, # doesn't matter for now, it's apples all the way dow
        ),
    ]

    # this should only ever be run on darwin anyway - sensors for visualization
    # SENSORS += [
    #     RealRGBSensorStretchIntel(
    #         height=desired_screen_size,
    #         width=desired_screen_size,
    #         use_resnet_normalization=True,
    #         mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
    #         stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
    #         uuid="rgb_lowres_only_viz",
    #     ),
    #     RealRGBSensorStretchKinect(
    #         height=desired_screen_size,
    #         width=desired_screen_size,
    #         use_resnet_normalization=True,
    #         mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
    #         stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
    #         uuid="rgb_lowres_arm_only_viz",
    #     ),
    # ]

    MAX_STEPS = 60

    TASK_SAMPLER = RealStretchAllRoomsObjectNavTaskSampler #RealStretchDiverseBringObjectTaskSampler
    TASK_TYPE = RealStretchObjectNavTask #RealStretchExploreWiseRewardTask
    ENVIRONMENT_TYPE = StretchRealEnvironment # account for the super init
    VISUALIZE = True

    NUM_PROCESSES = 1
    NUMBER_OF_TEST_PROCESS = 0
    NUMBER_OF_VALID_PROCESS = 0
    VAL_DEVICES = []
    TEST_DEVICES = []

    TRAIN_SCENES = ['RealRobothor']
    TEST_SCENES = ['RealRobothor']
