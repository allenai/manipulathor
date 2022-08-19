import platform
import random
import yaml
import torch

from utils.stretch_utils.stretch_thor_sensors import RGBSensorStretchKinect, RGBSensorStretchIntel
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.stretch_object_nav_baselines.experiments.ithor.obj_nav_2camera_ithor_wide import \
     ithorObjectNavClipResnet50RGBOnly2CameraWideFOV
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor

from utils.stretch_utils.stretch_object_nav_tasks import \
    StretchObjectNavTaskSegmentationSuccessActionFail, ExploreWiseObjectNavTask, StretchObjectNavTaskIntelSegmentationSuccess
from utils.stretch_utils.all_rooms_object_nav_task_sampler import RoboTHORObjectNavTaskSampler
from manipulathor_baselines.stretch_object_nav_baselines.models.clip_resnet_ncamera_preprocess_mixin \
    import TaskIdSensor

from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS, ROBOTHOR_TRAIN, ROBOTHOR_VAL



class RobothorObjectNavClipResnet50RGBOnly2CameraNarrowFOV(
    ithorObjectNavClipResnet50RGBOnly2CameraWideFOV
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    TRAIN_SCENES = ROBOTHOR_TRAIN
    TEST_SCENES = ROBOTHOR_VAL

    with open('datasets/objects/robothor_habitat2022.yaml', 'r') as f:
        OBJECT_TYPES=yaml.safe_load(f)

    random.shuffle(TRAIN_SCENES)
    random.shuffle(TEST_SCENES)

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


    TASK_SAMPLER = RoboTHORObjectNavTaskSampler
    TASK_TYPE = StretchObjectNavTaskIntelSegmentationSuccess
    POTENTIAL_VISUALIZERS = []
    
    MAX_STEPS = 300

    @classmethod
    def tag(cls):
        return cls.TASK_TYPE.__name__ + '-RGB-2Camera-RoboTHOR-Narrow' + '-' +  cls.WHICH_AGENT
    
    NUM_PROCESSES = 40 # one them crashed for space?
    def __init__(
            self,
            distributed_nodes: int = 1,
    ):
        super().__init__()
        self.distributed_nodes = distributed_nodes
        self.train_gpu_ids = tuple(range(torch.cuda.device_count())) # should I do this for everyone?, should i add val

    def machine_params(self, mode="train", **kwargs):
        params = super().machine_params(mode, **kwargs)

        if mode == "train":
            params.devices = params.devices * self.distributed_nodes
            params.nprocesses = params.nprocesses * self.distributed_nodes
            params.sampler_devices = params.sampler_devices * self.distributed_nodes

            if "machine_id" in kwargs:
                machine_id = kwargs["machine_id"]
                assert (
                        0 <= machine_id < self.distributed_nodes
                ), f"machine_id {machine_id} out of range [0, {self.distributed_nodes - 1}]"

                local_worker_ids = list(
                    range(
                        len(self.train_gpu_ids) * machine_id,
                        len(self.train_gpu_ids) * (machine_id + 1),
                        )
                )

                params.set_local_worker_ids(local_worker_ids)

            # Confirm we're setting up train params nicely:
            print(
                f"devices {params.devices}"
                f"\nnprocesses {params.nprocesses}"
                f"\nsampler_devices {params.sampler_devices}"
                f"\nlocal_worker_ids {params.local_worker_ids}"
            )
        elif mode == "valid":
            # Use all GPUs at their maximum capacity for training
            # (you may run validation in a separate machine)
            params.nprocesses = (0,)

        return params


    
