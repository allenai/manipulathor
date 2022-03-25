"""Task Samplers for the task of ArmPointNav"""
import glob
import json
import os
import random
from typing import List, Dict, Optional, Any, Union

from PIL import Image
import torchvision.transforms as transforms
import gym
import torch
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed

from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import BringObjectTask
from ithor_arm.ithor_arm_constants import transport_wrapper
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_tasks import (
    AbstractPickUpDropOffTask,
)
from ithor_arm.ithor_arm_viz import LoggerVisualizer, BringObjImageVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from utils.stretch_utils.real_stretch_environment import StretchRealEnvironment
from utils.stretch_utils.stretch_bring_object_task_samplers import StretchDiverseBringObjectTaskSampler


class RealStretchDiverseBringObjectTaskSampler(StretchDiverseBringObjectTaskSampler):

    def _create_environment(self, **kwargs) -> ManipulaTHOREnvironment:
        env = StretchRealEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )
        return env

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self.all_possible_points = {}

        if self.sampler_mode == "test":
            self.max_tasks = self.reset_tasks = 200


    def next_task(
            self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        if self.env is None:
            self.env = self._create_environment()
        self.env.reset(scene_name='RealRobothor')
        #TODO change these later
        init_object = {'object_id': 'Apple|1|1|1'}
        goal_object = {'object_id': 'Mug|1|1|1'}


        task_info = {
            'source_object_id': init_object['object_id'],
            'goal_object_id': goal_object['object_id'],
            "init_location": {},
            "goal_location": {},
            'agent_initial_state': {},
            'initial_object_location':{},
            'initial_hand_state': {},
            'episode_number': random.uniform(0, 10000),
            'scene_name':'RealRobothor'
        }


        self._last_sampled_task = self.TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task