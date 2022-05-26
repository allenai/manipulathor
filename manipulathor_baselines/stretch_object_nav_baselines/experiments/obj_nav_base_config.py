from typing import Sequence

import torch
import torch.optim as optim

# from ai2thor.platform import CloudRendering
# from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_thor_base import BringObjectThorBaseConfig
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    TrainingSettings,
)
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig

from allenact.base_abstractions.sensor import Sensor

from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from ithor_arm.ithor_arm_viz import TestMetricLogger

from utils.stretch_utils.all_rooms_object_nav_task_sampler import AllRoomsObjectNavTaskSampler
from utils.stretch_utils.stretch_object_nav_tasks import ObjectNavTask
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID


class ObjectNavBaseConfig(BringObjectThorBaseConfig):
    """The base config for ProcTHOR ObjectNav experiments."""

    TASK_SAMPLER = AllRoomsObjectNavTaskSampler
    TASK_TYPE = ObjectNavTask

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


    POTENTIAL_VISUALIZERS = [StretchObjNavImageVisualizer, TestMetricLogger]

    DISTANCE_TYPE = "l2"  # "geo"  # Can be "geo" or "l2"

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "reached_horizon_reward": 0.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0, 
            "positive_only_reward": False,
        }
        

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

