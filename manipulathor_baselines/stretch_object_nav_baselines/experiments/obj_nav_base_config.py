from abc import ABC
import platform
from math import ceil
from typing import Optional, Sequence, Dict, Any, List, Union

import gym
import numpy as np
import torch
import torch.optim as optim

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.task import TaskSampler
import ai2thor.fifo_server

from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    TrainingSettings,
    evenly_distribute_count_into_bins
)
from allenact.base_abstractions.experiment_config import MachineParams
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, ExpertActionSensor

from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from ithor_arm.ithor_arm_viz import TestMetricLogger

from utils.stretch_utils.all_rooms_object_nav_task_sampler import AllRoomsObjectNavTaskSampler
from utils.stretch_utils.stretch_object_nav_tasks import ObjectNavTask
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID, STRETCH_ENV_ARGS, UPDATED_PROCTHOR_COMMIT_ID

LOCOBOT_ENV_ARGS = dict(
    gridSize=0.25,
    width=224,
    height=224,
    visibilityDistance=1.0,
    agentMode="locobot",
    fieldOfView=63.453048374758716,
    rotateStepDegrees=30.0,
    snapToGrid=False,
    renderInstanceSegmentation=False,
    renderDepthImage=False,
    commit_id=UPDATED_PROCTHOR_COMMIT_ID
)

STRETCH_ENV_ARGS = dict(
    gridSize=0.25,
    width=224,
    height=224,
    visibilityDistance=3.0,
    agentMode='stretch',
    fieldOfView=69,
    agentControllerType="mid-level",
    server_class=ai2thor.fifo_server.FifoServer,
    snapToGrid=False,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=False,
    renderDepthImage=False,
    commit_id=UPDATED_PROCTHOR_COMMIT_ID, #### TODO not sure if this works for stretch
    horizon_init=0
)


class ObjectNavBaseConfig(ExperimentConfig, ABC):
    """The base config for ProcTHOR ObjectNav experiments."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    SENSORS: Optional[Sequence[Sensor]] = None

    # STEP_SIZE = 0.25
    # ROTATION_DEGREES = 30.0
    # VISIBILITY_DISTANCE = 1.0
    # STOCHASTIC = False

    VISUALIZE = False
    if platform.system() == "Darwin":
        VISUALIZE = True

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
    distributed_nodes = 1

    NUM_PROCESSES: Optional[int] = None
    NUMBER_OF_TEST_PROCESS: Optional[int] = None
    TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
    SAMPLER_GPU_IDS = TRAIN_GPU_IDS
    VALID_GPU_IDS = []
    TEST_GPU_IDS = []

    TRAIN_SCENES: str = None
    VALID_SCENES: str = None
    TEST_SCENES: str = None
    VALID_SAMPLES_IN_SCENE = 1
    TEST_SAMPLES_IN_SCENE = 1
    OBJECT_TYPES: Optional[Sequence[str]] = None

    WHICH_AGENT = None

    CAMERA_WIDTH = 224
    CAMERA_HEIGHT = 224
    SCREEN_SIZE = 224

    POTENTIAL_VISUALIZERS = [StretchObjNavImageVisualizer, TestMetricLogger]

    TASK_SAMPLER = AllRoomsObjectNavTaskSampler
    TASK_TYPE = ObjectNavTask
    DISTANCE_TYPE = "l2"  # "geo"  # Can be "geo" or "l2"
    MAX_STEPS = 500

    def __init__(self):
        # super().__init__()
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "reached_horizon_reward": 0.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0, 
            "positive_only_reward": False,
        } # shaping weight removed for eval in task sampler
        if self.WHICH_AGENT == 'locobot':
            self.ENV_ARGS = LOCOBOT_ENV_ARGS
        elif self.WHICH_AGENT == 'stretch':
            self.ENV_ARGS = STRETCH_ENV_ARGS
        else:
            raise NotImplementedError

    
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        from datetime import datetime

        now = datetime.now()

        exp_name_w_time = cls.__name__ + "_" + now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        if cls.VISUALIZE:
            visualizers = [
                viz(exp_name=exp_name_w_time) for viz in cls.POTENTIAL_VISUALIZERS
            ]
            kwargs["visualizers"] = visualizers
        kwargs["objects"] = cls.OBJECT_TYPES
        kwargs["task_type"] = cls.TASK_TYPE
        kwargs["exp_name"] = exp_name_w_time
        return cls.TASK_SAMPLER(**kwargs)

    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        devices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        out = {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(self.TASK_TYPE.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
        }
        
        x_display = (("0.%d" % devices[process_ind % len(devices)]) if len(devices) > 0 else None)
        out['env_args']["x_display"] = x_display
        return out

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            devices=devices
        )
        res["scene_period"] = "manual"
        res["sampler_mode"] = "train"

        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.VALID_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            devices=devices
        )
        res["scene_period"] = self.VALID_SAMPLES_IN_SCENE
        res["sampler_mode"] = "val"
        res["max_tasks"] = self.VALID_SAMPLES_IN_SCENE * len(res["scenes"])
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TEST_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            devices=devices
        )
        res["scene_period"] = self.TEST_SAMPLES_IN_SCENE
        res["sampler_mode"] = "test"
        return res
        

    def training_pipeline(self, **kwargs):
        log_interval_small = (
            self.distributed_nodes*self.NUM_PROCESSES * 32 * 10
            if torch.cuda.is_available
            else 1
        )
        log_interval_medium = (
            self.distributed_nodes*self.NUM_PROCESSES * 64 * 5
            if torch.cuda.is_available
            else 1
        )
        log_interval_large = (
            self.distributed_nodes*self.NUM_PROCESSES * 128 * 5
            if torch.cuda.is_available
            else 1
        )

        batch_steps_0 = int(10e6)
        batch_steps_1 = int(10e6)
        batch_steps_2 = int(1e9) - batch_steps_1 - batch_steps_0

        return TrainingPipeline(
            save_interval=2_000_000,
            metric_accumulate_interval=10_000,
            optimizer_builder=Builder(optim.Adam, dict(lr=0.0003)),
            num_mini_batch=1,
            update_repeats=4,
            max_grad_norm=0.5,
            num_steps=128,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps_0,
                    training_settings=TrainingSettings(
                        num_steps=32, metric_accumulate_interval=log_interval_small
                    ),
                ),
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps_1,
                    training_settings=TrainingSettings(
                        num_steps=64, metric_accumulate_interval=log_interval_medium
                    ),
                ),
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps_2,
                    training_settings=TrainingSettings(
                        num_steps=128, metric_accumulate_interval=log_interval_large
                    ),
                ),
            ],
        )
    
    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[int] = []
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAIN_GPU_IDS * workers_per_device
            )
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else evenly_distribute_count_into_bins(self.NUM_PROCESSES, len(gpu_ids))
            )
            sampler_devices = self.SAMPLER_GPU_IDS
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALID_GPU_IDS
        elif mode == "test":
            # nprocesses = self.NUMBER_OF_TEST_PROCESS if torch.cuda.is_available() else 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TEST_GPU_IDS
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else evenly_distribute_count_into_bins(self.NUMBER_OF_TEST_PROCESS, len(gpu_ids))
            )

            # print('Test Mode', gpu_ids, 'because cuda is',torch.cuda.is_available(), 'number of workers', nprocesses)
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

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

        # remove
        # print('MACHINE PARAM', 'nprocesses',nprocesses,'devices',gpu_ids,'sampler_devices',sampler_devices,'mode = train',  mode == "train",'gpu ids', gpu_ids,)  # ignored with > 1 gpu_ids)
        return MachineParams(nprocesses=nprocesses,
        devices=gpu_ids,
        sampler_devices=sampler_devices if mode == "train" else gpu_ids,  # ignored with > 1 gpu_ids
        sensor_preprocessor_graph=sensor_preprocessor_graph)


