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

from ithor_arm.arm_calculation_utils import initialize_arm
from ithor_arm.bring_object_tasks import BringObjectTask, WPickUpBringObjectTask, WPickUPExploreBringObjectTask, NoPickUPExploreBringObjectTask
from ithor_arm.explore_environment import ExploreEnvironment
from ithor_arm.ithor_arm_constants import transport_wrapper
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_tasks import (
    AbstractPickUpDropOffTask,
)
from ithor_arm.ithor_arm_viz import LoggerVisualizer, BringObjImageVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.jupyter_helper import get_reachable_positions
from utils.manipulathor_data_loader_utils import get_random_query_image


class ExploreTaskSampler(TaskSampler):

    # _TASK_TYPE = Task

    def __init__(
        self,
        scenes: List[str],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        objects: List[str],
        task_type: type,
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        fixed_tasks: Optional[List[Dict[str, Any]]] = None,
        visualizers: List[LoggerVisualizer] = [],
        *args,
        **kwargs
    ) -> None:
        self.TASK_TYPE = task_type
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.grid_size = 0.25
        self.env: Optional[ManipulaTHOREnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.objects = objects

        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        self.scene_period: Optional[
            Union[str, int]
        ] = scene_period  # default makes a random choice
        self.max_tasks: Optional[int] = None
        self.reset_tasks = max_tasks

        self._last_sampled_task: Optional[Task] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()
        self.visualizers = visualizers
        self.sampler_mode = kwargs["sampler_mode"]
        self.cap_training = kwargs["cap_training"]

    def _create_environment(self, **kwargs) -> ManipulaTHOREnvironment:
        env = ExploreEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )

        return env

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.sampler_index = 0

        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return (
            self.total_unique - self.sampler_index
            if self.sampler_mode != "train"
            else (float("inf") if self.max_tasks is None else self.max_tasks)
        )
    def reset_scene(self, scene_name):
        self.env.reset(
            scene_name=scene_name#, agentMode="arm", agentControllerType="mid-level"
        )
        #NOTE @samir you can use or not use the following up to you
        self.env.step(dict(action="RandomizeMaterials"))
        self.env.step(dict(action="RandomizeLighting"))



    def next_task(
            self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:

        if self.env is None:
            self.env = self._create_environment()

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        # ForkedPdb().set_trace()
        if self.sampler_mode != "train" and self.length <= 0:
            return None

        scene_name = random.choice(self.scenes)
        self.reset_scene(scene_name)
        possible_agent_locations_in_scene = get_reachable_positions(self.env.controller)
        agent_state = random.choice(possible_agent_locations_in_scene)

        this_controller = self.env


        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["x"],
                y=agent_state["y"],
                z=agent_state["z"],
                rotation=dict(
                    x=0,
                    y=random.choice([0, 90, 180, 270]), #NOTE @samir
                    z=0,
                ),
                horizon=random.choice([-30, 0, 30]),
            )
        )

        if not event.metadata['lastActionSuccess']:
            print('ERROR: Teleport failed')



        task_info = {
            # 'source_object_id': init_object['object_id'],
            # 'goal_object_id': goal_object['object_id'],
            # "init_location": init_object,
            # "goal_location": goal_object,
            # 'agent_initial_state': initial_agent_location,
            # 'initial_object_location':initial_object_info,
            # 'initial_hand_state': initial_hand_state,
            # 'source_object_query': source_object_query,
            # 'goal_object_query': goal_object_query,
            'episode_number': random.uniform(0, 10000),
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

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        if self.sampler_mode == "train":
            return None
        else:
            if self.max_tasks is None:
                self.max_tasks = 100

            return min(self.max_tasks, 1000)
