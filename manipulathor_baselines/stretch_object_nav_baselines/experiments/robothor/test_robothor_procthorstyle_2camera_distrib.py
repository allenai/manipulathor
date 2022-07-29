import datasets
import torch
from typing import Sequence
from typing_extensions import Literal

from manipulathor_baselines.stretch_object_nav_baselines.experiments.procthor.obj_nav_2camera_procthor_narrow \
    import ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV

from utils.procthor_utils.procthor_object_nav_task_samplers import RoboThorObjectNavTestTaskSampler
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from manipulathor_utils.debugger_util import ForkedPdb

import prior

from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.base_abstractions.sensor import (
    ExpertActionSensor,
    SensorSuite,
)

class ObjectNavRoboTHORTestProcTHORstyle(ProcTHORObjectNavClipResnet50RGBOnly2CameraNarrowFOV):

    EVAL_TASKS = datasets.load_dataset(
        f"allenai/robothor-objectnav-eval", use_auth_token=True
    )
    # EVAL_TASKS = prior.load_dataset("object-nav-eval")


    TEST_TASK_SAMPLER = RoboThorObjectNavTestTaskSampler
    TEST_ON_VALIDATION = True
    # TEST_GPU_IDS = list(range(torch.cuda.device_count())) # uncomment for vision server testing

    # NUM_PROCESSES = 56 # one them crashed for space?
    # NUM_TRAIN_PROCESSES = 64
    # NUM_VAL_PROCESSES = 2
    # NUM_TEST_PROCESSES = 60

    NUM_TRAIN_PROCESSES = 0
    NUM_TEST_PROCESSES = 1
    NUM_VAL_PROCESSES = 0

    TRAIN_DEVICES = (
        tuple(range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else (torch.device("cpu"),)
    )
    VAL_DEVICES = (
        (torch.cuda.device_count() - 1,)
        if torch.cuda.is_available()
        else (torch.device("cpu"),)
    )
    TEST_DEVICES = (
        tuple(range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else (torch.device("cpu"),)
    )
    def __init__(
            self,
            distributed_nodes: int = 1,
    ):
        super().__init__()
        self.distributed_nodes = distributed_nodes

    @classmethod
    def tag(cls):
        return super().tag() + "-RoboTHOR-Test"
    
    @classmethod
    def make_sampler_fn(cls, **kwargs):
        from datetime import datetime

        now = datetime.now()

        exp_name_w_time = cls.__name__ + "_" + now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        if cls.VISUALIZE:
            visualizers = [
                viz(exp_name=exp_name_w_time) for viz in cls.POTENTIAL_VISUALIZERS
            ]
            kwargs["visualizers"] = visualizers
        kwargs["exp_name"] = exp_name_w_time

        if kwargs["sampler_mode"] == "train":
            return cls.TASK_SAMPLER(**kwargs)
        else:
            return cls.TEST_TASK_SAMPLER(**kwargs)

    def valid_task_sampler_args(self, **kwargs):
        out = self._get_sampler_args_for_scene_split(
            houses=self.EVAL_TASKS["validation"].shuffle(),
            mode="eval",
            max_tasks=20,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            **kwargs,
        )
        return out

    def test_task_sampler_args(self, **kwargs):
        if self.TEST_ON_VALIDATION:
            houses = self.EVAL_TASKS["validation"]
        else:
            houses = self.EVAL_TASKS["test"].shuffle()
            # return self.valid_task_sampler_args(**kwargs)

        out = self._get_sampler_args_for_scene_split(
            houses=houses,
            mode="eval",
            max_tasks=None,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,  # ignored
            **kwargs,
        )
        return out
    
    def machine_params(self, mode: Literal["train", "valid", "test"], **kwargs):
        devices: Sequence[torch.device]
        nprocesses: int
        if mode == "train":
            devices = self.TRAIN_DEVICES * self.distributed_nodes
            nprocesses = self.NUM_TRAIN_PROCESSES * self.distributed_nodes
        elif mode == "valid":
            devices = self.VAL_DEVICES
            nprocesses = self.NUM_VAL_PROCESSES
        elif mode == "test":
            devices = self.TEST_DEVICES
            nprocesses = self.NUM_TEST_PROCESSES

        nprocesses = evenly_distribute_count_into_bins(
            count=nprocesses, nbins=len(devices)
        )

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        params = MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=devices,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

        # NOTE: for distributed setup
        if mode == "train" and "machine_id" in kwargs:
            machine_id = kwargs["machine_id"]
            assert (
                0 <= machine_id < self.distributed_nodes
            ), f"machine_id {machine_id} out of range [0, {self.distributed_nodes} - 1]"
            local_worker_ids = list(
                range(
                    len(self.TRAIN_DEVICES) * machine_id,
                    len(self.TRAIN_DEVICES) * (machine_id + 1),
                )
            )
            params.set_local_worker_ids(local_worker_ids)

        return params