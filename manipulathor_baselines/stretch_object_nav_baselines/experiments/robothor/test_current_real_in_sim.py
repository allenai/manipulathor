import platform
import random
import yaml

from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.stretch_object_nav_baselines.experiments.ithor.obj_nav_2camera_ithor_wide import \
     ithorObjectNavClipResnet50RGBOnly2CameraWideFOV
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor

from utils.stretch_utils.stretch_object_nav_tasks import \
    StretchObjectNavTaskIntelSegmentationSuccess, StretchObjectNavTaskSegmentationSuccessActionFail, ExploreWiseObjectNavTask
from utils.stretch_utils.all_rooms_object_nav_task_sampler import RealSimRealObjNavSampler

# from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS, ROBOTHOR_TRAIN, ROBOTHOR_VAL
from manipulathor_baselines.stretch_object_nav_baselines.models.clip_resnet_ncamera_preprocess_mixin \
    import TaskIdSensor


class RobothorObjectNavClipResnet50RGBOnly2CameraNarrowFOV(
    ithorObjectNavClipResnet50RGBOnly2CameraWideFOV
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    TRAIN_SCENES = ['FloorPlan_RoboTHOR_Real']
    # TEST_SCENES = ROBOTHOR_VAL
    # OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list if room_typ == 'robothor']))
    # OBJECT_TYPES.sort()

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    SENSORS = [
        RGBSensorStretchIntel(
            height=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            width=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        RGBSensorStretchKinect(
            height=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            width=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
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

    TASK = StretchObjectNavTaskIntelSegmentationSuccess
    TASK_SAMPLER = RealSimRealObjNavSampler