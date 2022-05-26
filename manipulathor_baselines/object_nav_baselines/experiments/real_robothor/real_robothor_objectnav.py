from utils.stretch_utils.stretch_constants import INTEL_CAMERA_WIDTH
from manipulathor_utils.debugger_util import ForkedPdb

from utils.stretch_utils.real_stretch_sensors import RealRGBSensorStretchIntel, RealRGBSensorStretchKinect
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from utils.procthor_utils.all_rooms_object_nav_task_sampler import AllRoomsObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import StretchObjectNavTask


from manipulathor_baselines.object_nav_baselines.experiments.ithor.obj_nav_2camera_ithor_wide import \
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
            uuid="rgb_lowres",
        ),
        RealRGBSensorStretchKinect(
            height=desired_screen_size,
            width=desired_screen_size,
            use_resnet_normalization=True,
            uuid="rgb_lowres_arm",
        ),
        GoalObjectTypeThorSensor(
            object_types=ithorObjectNavClipResnet50RGBOnly2CameraWideFOV.OBJECT_TYPES, # doesn't matter for now, it's apples all the way dow
        ),
    ]

    MAX_STEPS = 200

    TASK_SAMPLER = AllRoomsObjectNavTaskSampler#RealStretchDiverseBringObjectTaskSampler
    TASK_TYPE = StretchObjectNavTask#RealStretchExploreWiseRewardTask

    NUM_PROCESSES = 20


    TRAIN_SCENES = ['RealRobothor']
    TEST_SCENES = ['RealRobothor']
