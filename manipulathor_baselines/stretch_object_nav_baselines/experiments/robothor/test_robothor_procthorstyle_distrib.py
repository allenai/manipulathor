from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Literal
import torch

from manipulathor_baselines.stretch_object_nav_baselines.experiments.robothor.test_robothor_procthorstyle import \
    ObjectNavRoboTHORTestProcTHORstyle
from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.base_abstractions.sensor import (
    ExpertActionSensor,
    SensorSuite,
)


class ObjectNavRoboTHORTestProcTHORstyleDistrib(
    ObjectNavRoboTHORTestProcTHORstyle
):
    NUM_PROCESSES = 56 # one them crashed for space?
    NUM_TRAIN_PROCESSES = 64
    NUM_VAL_PROCESSES = 8
    NUM_TEST_PROCESSES = 0

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
        # self.train_gpu_ids = tuple(range(torch.cuda.device_count())) # should I do this for everyone?, should i add val


    # def machine_params(self, mode="train", **kwargs):
    #     params = super().machine_params(mode, **kwargs)

    #     if mode == "train":
    #         params.devices = params.devices * self.distributed_nodes
    #         params.nprocesses = params.nprocesses * self.distributed_nodes
    #         params.sampler_devices = params.sampler_devices * self.distributed_nodes

    #         if "machine_id" in kwargs:
    #             machine_id = kwargs["machine_id"]
    #             assert (
    #                     0 <= machine_id < self.distributed_nodes
    #             ), f"machine_id {machine_id} out of range [0, {self.distributed_nodes - 1}]"

    #             local_worker_ids = list(
    #                 range(
    #                     len(self.train_gpu_ids) * machine_id,
    #                     len(self.train_gpu_ids) * (machine_id + 1),
    #                     )
    #             )

    #             params.set_local_worker_ids(local_worker_ids)

    #         # Confirm we're setting up train params nicely:
    #         print(
    #             f"devices {params.devices}"
    #             f"\nnprocesses {params.nprocesses}"
    #             f"\nsampler_devices {params.sampler_devices}"
    #             f"\nlocal_worker_ids {params.local_worker_ids}"
    #         )
    #     elif mode == "valid":
    #         # Use all GPUs at their maximum capacity for training
    #         # (you may run validation in a separate machine)
    #         params.nprocesses = (0,)

    #     return params 

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