from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Literal, final
import platform

import datasets
import numpy as np
from math import ceil
import torch

from allenact.utils.system import get_logger
from allenact.base_abstractions.preprocessor import (
    Preprocessor,
    SensorPreprocessorGraph,
)
from allenact.base_abstractions.sensor import (
    Sensor,
    Union,
)
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.experiment_utils import (
    Builder,
)

from manipulathor_baselines.procthor_baselines.experiments.ithor.obj_nav_base_config import ObjectNavBaseConfig
from utils.procthor_utils.procthor_object_nav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import ObjectNavTask
from utils.procthor_utils.procthor_helper import PROCTHOR_INVALID_SCENES
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID

from manipulathor_utils.debugger_util import ForkedPdb



class ProcTHORObjectNavBaseConfig(ObjectNavBaseConfig):
    """The base config for ProcTHOR ObjectNav experiments."""

    TASK_SAMPLER = ProcTHORObjectNavTaskSampler
    TASK_TYPE = ObjectNavTask

    OBJECT_TYPES: Optional[Sequence[str]]

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

    SENSORS: Sequence[Sensor] = []

    DISTANCE_TYPE = "l2"  # "geo"  # Can be "geo" or "l2"

    CAP_TRAINING = None

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[
        int
    ] = 20 # default config/main.yaml
    RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = (
        -1
    )  # Should be > 0 if `ADVANCE_SCENE_ROLLOUT_PERIOD` is `None`

    if platform.system() == "Darwin":
        RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = 1

    RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 1 # TODO apparently this won't work with 100 (why?)

    HOUSE_DATASET = datasets.load_dataset(
        "allenai/houses", use_auth_token=True, ignore_verifications=True
    )

    MAX_STEPS = 500
    NUM_TRAIN_HOUSES = None # none means all


    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()

    @classmethod
    def make_sampler_fn(
        cls, **kwargs
    ) -> TaskSampler:
        from datetime import datetime

        now = datetime.now()

        exp_name_w_time = cls.__name__ + "_" + now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        if cls.VISUALIZE:
            visualizers = [
                viz(exp_name=exp_name_w_time) for viz in cls.POTENTIAL_VISUALIZERS
            ]
            kwargs["visualizers"] = visualizers
        kwargs["task_type"] = cls.TASK_TYPE
        kwargs["exp_name"] = exp_name_w_time

        return cls.TASK_SAMPLER(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(
            np.linspace(start=0, stop=n, num=num_parts + 1, endpoint=True)
        ).astype(np.int32)

    def _get_sampler_args_for_scene_split(
        self,
        houses: datasets.Dataset,
        mode: Literal["train", "eval"],
        resample_same_scene_freq: int,
        allow_oversample: bool,
        process_ind: int,
        total_processes: int,
        max_tasks: Optional[int],
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        # NOTE: oversample some scenes -> bias
        oversample_warning = (
            f"Warning: oversampling some of the houses ({houses}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        house_inds = list(range(len(houses)))
        valid_house_inds = [h for h in house_inds if h not in PROCTHOR_INVALID_SCENES]
        if total_processes > len(valid_house_inds):
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(houses)`"
                    f" ({total_processes} > {len(houses)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(houses) != 0:
                get_logger().warning(oversample_warning)
            valid_house_inds = valid_house_inds * ceil(total_processes / len(houses))
            valid_house_inds = valid_house_inds[
                : total_processes * (len(valid_house_inds) // total_processes)
            ]
        elif len(houses) % total_processes != 0:
            get_logger().warning(oversample_warning)

        inds = self._partition_inds(len(valid_house_inds), total_processes)
        selected_house_inds = valid_house_inds[inds[process_ind] : inds[process_ind + 1]]
        
        x_display = (("0.%d" % devices[process_ind % len(devices)]) if len(devices) > 0 else None)

        return {
            "sampler_mode": mode,
            "houses": houses,
            "house_inds": selected_house_inds,
            "process_ind": process_ind,
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
            "target_object_types": self.OBJECT_TYPES,
            "max_tasks": max_tasks if max_tasks is not None else len(house_inds),
            "distance_type": self.DISTANCE_TYPE,
            "resample_same_scene_freq": resample_same_scene_freq,
            "scene_name": "Procedural"

        }, x_display

    def train_task_sampler_args(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        train_houses = self.HOUSE_DATASET["train"]
        if self.NUM_TRAIN_HOUSES:
            train_houses = train_houses.select(range(self.NUM_TRAIN_HOUSES))

        out,x_display = self._get_sampler_args_for_scene_split(
            houses=train_houses,
            mode="train",
            allow_oversample=True,
            max_tasks=float("inf"),
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN,
            **kwargs,
        )
        out["cap_training"] = self.CAP_TRAINING
        out["env_args"] = {}
        out["env_args"].update(self.ENV_ARGS)
        out["env_args"]["x_display"] = x_display
        out["env_args"]['commit_id'] = PROCTHOR_COMMIT_ID
        out["env_args"]['scene'] = 'Procedural'
        return out

    def valid_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        val_houses = self.HOUSE_DATASET["validation"]
        out,x_display = self._get_sampler_args_for_scene_split(
            houses=val_houses.select(range(100)),
            mode="eval",
            allow_oversample=False,
            max_tasks=10,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            **kwargs,
        )
        out["cap_training"] = self.CAP_TRAINING
        out["env_args"] = {}
        out["env_args"].update(self.ENV_ARGS)
        out["env_args"]['allow_flipping'] = False        
        out["env_args"]["x_display"] = x_display
        out["env_args"]['commit_id'] = PROCTHOR_COMMIT_ID
        out["env_args"]['scene'] = 'Procedural'
        return out

    def test_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        if self.TEST_ON_VALIDATION:
            return self.valid_task_sampler_args(**kwargs)

        test_houses = self.HOUSE_DATASET["test"]
        out,x_display = self._get_sampler_args_for_scene_split(
            houses=test_houses.select(range(100)),
            mode="eval",
            allow_oversample=False,
            max_tasks=10,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            **kwargs,
        )
        out["cap_training"] = self.CAP_TRAINING
        out["env_args"] = {}
        out["env_args"].update(self.ENV_ARGS)
        out["env_args"]['allow_flipping'] = False  
        out["env_args"]["x_display"] = x_display
        out["env_args"]['commit_id'] = PROCTHOR_COMMIT_ID
        out["env_args"]['scene'] = 'Procedural'
        return out


