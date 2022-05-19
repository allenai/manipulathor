from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Literal, final

import datasets
import numpy as np
import torch
import torch.optim as optim

from ai2thor.platform import CloudRendering
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
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

# from allenact.utils.system import get_logger
# from training import cfg
# from training.tasks.object_nav import ObjectNavTaskSampler
# from training.utils.types import RewardConfig, TaskSamplerArgs
from utils.procthor_utils.procthor_object_nav_task_samplers import ProcTHORObjectNavTaskSampler
from utils.procthor_utils.procthor_object_nav_tasks import ObjectNavTask
from utils.stretch_utils.stretch_constants import PROCTHOR_COMMIT_ID


class ProcTHORObjectNavBaseConfig(BringObjectThorBaseConfig):
    """The base config for ProcTHOR ObjectNav experiments."""

    TASK_SAMPLER = ProcTHORObjectNavTaskSampler
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
    ] = 20 # default config/main.yaml
    RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = (
        -1
    )  # Should be > 0 if `ADVANCE_SCENE_ROLLOUT_PERIOD` is `None`
    RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 1 # TODO apparently this won't work with 100 (why?)
    
    @final
    def training_pipeline(self, **kwargs):
        log_interval_small = (
            self.NUM_PROCESSES * 32 * 10
            if torch.cuda.is_available
            else 1
        )
        log_interval_medium = (
            self.NUM_PROCESSES * 64 * 5
            if torch.cuda.is_available
            else 1
        )
        log_interval_large = (
            self.NUM_PROCESSES * 128 * 5
            if torch.cuda.is_available
            else 1
        )

        batch_steps_0 = int(10e6)
        batch_steps_1 = int(10e6)
        batch_steps_2 = int(1e9) - batch_steps_1 - batch_steps_0

        return TrainingPipeline(
            save_interval=5_000_000,
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

