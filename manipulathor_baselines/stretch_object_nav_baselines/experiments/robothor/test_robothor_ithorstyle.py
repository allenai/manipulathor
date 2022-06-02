import platform
import random
import yaml

from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from utils.stretch_utils.stretch_object_nav_tasks import StretchNeckedObjectNavTask


from manipulathor_baselines.stretch_object_nav_baselines.experiments.ithor.obj_nav_2camera_ithor_wide import \
     ithorObjectNavClipResnet50RGBOnly2CameraWideFOV
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor

from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS, ROBOTHOR_TRAIN, ROBOTHOR_VAL



class RoboTHORObjectNavTestiTHORstyleSingleCam(
    ithorObjectNavClipResnet50RGBOnly2CameraWideFOV
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    TRAIN_SCENES = ROBOTHOR_TRAIN
    TEST_SCENES = ROBOTHOR_VAL
    # OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list if room_typ == 'robothor']))
    # OBJECT_TYPES.sort()

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    random.shuffle(TRAIN_SCENES)
    random.shuffle(TEST_SCENES)

    SENSORS = [
        RGBSensorThor(
            height=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
            width=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
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
            RGBSensorThor(
                height=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
                width=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.SCREEN_SIZE,
                use_resnet_normalization=True,
                mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
                stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
                uuid="rgb_lowres_only_viz",
            ),
        ]
    
    TASK_TYPE = StretchNeckedObjectNavTask
    MAX_STEPS = 200

    def __init__(self):
        super().__init__() 
        self.ENV_ARGS['agentMode']='default'
        # maybe - include updated procthor commit ID? check if it makes a difference





    
