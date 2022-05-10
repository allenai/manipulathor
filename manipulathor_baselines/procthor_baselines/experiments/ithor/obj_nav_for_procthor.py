from abc import ABC
from math import ceil
from typing import Any, Dict, List, Literal, Optional, Sequence, final

import datasets
import numpy as np
import torch
import torch.optim as optim

from ai2thor.platform import CloudRendering
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_thor_base import BringObjectThorBaseConfig

from allenact.base_abstractions.sensor import Sensor

from allenact.utils.system import get_logger
# from training import cfg
# from training.tasks.object_nav import ObjectNavTaskSampler
# from training.utils.types import RewardConfig, TaskSamplerArgs
from utils.procthor_utils.procthor_object_nav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import ProcTHORObjectNavTask
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID


class ProcTHORObjectNavBaseConfig(BringObjectThorBaseConfig, ABC):
    """The base config for ProcTHOR ObjectNav experiments."""

    TASK_SAMPLER = ProcTHORObjectNavTaskSampler
    TASK_TYPE = ProcTHORObjectNavTask

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

    # TEST_ON_VALIDATION: bool = cfg.evaluation.test_on_validation

    # AGENT_MODE = cfg.agent.agent_mode
    # CAMERA_WIDTH = cfg.agent.camera_width
    # CAMERA_HEIGHT = cfg.agent.camera_height
    # STEP_SIZE = cfg.agent.step_size
    # VISIBILITY_DISTANCE = cfg.agent.visibility_distance
    # FIELD_OF_VIEW = cfg.agent.field_of_view
    # ROTATE_STEP_DEGREES = cfg.agent.rotate_step_degrees

    HOUSE_DATASET = datasets.load_dataset(
        "allenai/houses", use_auth_token=True, ignore_verifications=True
    )

    SENSORS: Sequence[Sensor] = []

    DISTANCE_TYPE = "l2"  # "geo"  # Can be "geo" or "l2"

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[
        int
    ] = cfg.training.advance_scene_rollout_period
    RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = (
        -1
    )  # Should be > 0 if `ADVANCE_SCENE_ROLLOUT_PERIOD` is `None`
    RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 100

    @staticmethod
    def get_platform(
        gpu_index: int, platform: Literal["CloudRendering", "Linux64"]
    ) -> Dict[str, Any]:
        """Return the platform specific args to be passed into AI2-THOR.
        Parameters:
        - gpu_index: The index of the GPU to use. Must be in the range [0,
          torch.cuda.device_count() - 1].
        """
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"gpu_index must be in the range [0, {torch.cuda.device_count()}]."
                f" You gave {gpu_index}."
            )

        if platform == "CloudRendering":
            # NOTE: There is an off-by-1 error with cloud rendering where
            # gpu_index cannot be set to 1. It maps 0=>0, 2=>1, 3=>2, etc.
            if gpu_index > 0:
                gpu_index += 1
            return {"gpu_device": gpu_index, "platform": CloudRendering}
        elif platform == "Linux64":
            return {"x_display": f":0.{gpu_index}"}
        else:
            raise ValueError(f"Unknown platform: {platform}")

    def _get_sampler_args_for_scene_split(
        self,
        houses: datasets.Dataset,
        mode: Literal["train", "eval"],
        resample_same_scene_freq: int,
        allow_oversample: bool,
        allow_flipping: bool,
        process_ind: int,
        total_processes: int,
        max_tasks: Optional[int],
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        extra_controller_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # NOTE: oversample some scenes -> bias
        oversample_warning = (
            f"Warning: oversampling some of the houses ({houses}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        house_inds = list(range(len(houses)))
        if total_processes > len(houses):
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(houses)`"
                    f" ({total_processes} > {len(houses)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(houses) != 0:
                get_logger().warning(oversample_warning)
            house_inds = house_inds * ceil(total_processes / len(houses))
            house_inds = house_inds[
                : total_processes * (len(house_inds) // total_processes)
            ]
        elif len(houses) % total_processes != 0:
            get_logger().warning(oversample_warning)

        inds = self._partition_inds(len(house_inds), total_processes)
        house_inds = house_inds[inds[process_ind] : inds[process_ind + 1]]

        controller_args = {
            "branch": "nanna",
            "width": self.CAMERA_WIDTH,
            "height": self.CAMERA_HEIGHT,
            "rotateStepDegrees": self.ROTATE_STEP_DEGREES,
            "visibilityDistance": self.VISIBILITY_DISTANCE,
            "gridSize": self.STEP_SIZE,
            "agentMode": self.AGENT_MODE,
            "fieldOfView": self.FIELD_OF_VIEW,
            "snapToGrid": False,
            "renderDepthImage": any(isinstance(s, DepthSensor) for s in self.SENSORS),
            **self.get_platform(
                gpu_index=devices[process_ind % len(devices)],
                platform="Linux64",
            ),
        }
        if extra_controller_args:
            controller_args.update(extra_controller_args)

        return dict(
            process_ind=process_ind,
            mode=mode,
            house_inds=house_inds,
            houses=houses,
            sensors=self.SENSORS,
            controller_args=controller_args,
            target_object_types=self.OBJECT_TYPES,
            max_steps=self.MAX_STEPS,
            seed=seeds[process_ind] if seeds is not None else None,
            deterministic_cudnn=deterministic_cudnn,
            reward_config=self.REWARD_CONFIG,
            max_tasks=max_tasks if max_tasks is not None else len(house_inds),
            allow_flipping=allow_flipping,
            distance_type=self.DISTANCE_TYPE,
            resample_same_scene_freq=resample_same_scene_freq,
        )

    def train_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        train_houses = self.HOUSE_DATASET["train"]
        # if cfg.procthor.num_train_houses:
        #     train_houses = train_houses.select(range(cfg.procthor.num_train_houses))

        out = self._get_sampler_args_for_scene_split(
            houses=train_houses,
            mode="train",
            allow_oversample=True,
            max_tasks=float("inf"),
            allow_flipping=True,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN,
            extra_controller_args=dict(branch="nanna", scene="Procedural"),
            **kwargs,
        )
        return {"task_sampler_args": out}

    def valid_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        val_houses = self.HOUSE_DATASET["validation"]
        out = self._get_sampler_args_for_scene_split(
            houses=val_houses.select(range(100)),
            mode="eval",
            allow_oversample=False,
            max_tasks=10,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            extra_controller_args=dict(scene="Procedural"),
            **kwargs,
        )
        return {"task_sampler_args": out}

    def test_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        if self.TEST_ON_VALIDATION:
            return self.valid_task_sampler_args(**kwargs)

        test_houses = self.HOUSE_DATASET["test"]
        out = self._get_sampler_args_for_scene_split(
            houses=test_houses.select(range(100)),
            mode="eval",
            allow_oversample=False,
            max_tasks=10,
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            extra_controller_args=dict(scene="Procedural"),
            **kwargs,
        )
        return {"task_sampler_args": out}


