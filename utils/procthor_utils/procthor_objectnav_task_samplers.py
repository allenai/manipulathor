"""Task Samplers for the task of ArmPointNav"""
import glob
import json
import os
import platform
import random
from typing import Optional, List, Union, Dict, Any

import datasets
import gym
import torch
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler, Task
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger
from numba.core.serialize import pickle
from omegaconf import OmegaConf
from torch.distributions.utils import lazy_property

from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import ROBOTHOR_TRAIN, KITCHEN_TRAIN, KITCHEN_TEST, KITCHEN_VAL
from utils.manipulathor_data_loader_utils import get_random_query_image, get_random_query_feature_from_img_adr
from utils.procthor_utils.procthor_types import AgentPose, Vector3
from utils.stretch_utils.stretch_constants import ADITIONAL_ARM_ARGS
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from scripts.stretch_jupyter_helper import get_reachable_positions, transport_wrapper, AGENT_ROTATION_DEG, \
    make_all_objects_unbreakable
from utils.stretch_utils.stretch_visualizer import StretchBringObjImageVisualizer

class ProcTHORObjectNavTaskSampler(TaskSampler):

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


    def _create_environment(self, **kwargs) -> StretchManipulaTHOREnvironment:
        env = self.environment_type(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )

        return env
    def set_reachable_positions(self):
        pass
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
        self.environment_type = env_args['environment_type']
        del env_args['environment_type']
        self.first_camera_horizon = kwargs['first_camera_horizon']
        self.env_args = env_args
        self.scenes = scenes
        self.grid_size = 0.25
        self.env: Optional[StretchManipulaTHOREnvironment] = None
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





        self.house_dataset = datasets.load_dataset("allenai/houses", use_auth_token=True)

        RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = (
            -1
        )  # Should be > 0 if `ADVANCE_SCENE_ROLLOUT_PERIOD` is `None`
        # RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 100
        # if platform.system() == "Darwin":
        #     RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 1

        RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 1
        self.resample_same_scene_freq = RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE
        assert self.resample_same_scene_freq == 1 # IMPORTANT IT WON"T WORK FOR 100
        self.episode_index = 0
        self.house_inds_index = 0
        self.reachable_positions_map = {}
        self.house_dataset = self.house_dataset['train'] #TODO separately for test and val

        ROOMS_TO_USE = [int(scene.replace('ProcTHOR', '')) for scene in self.scenes]


        self.args_house_inds = list(ROOMS_TO_USE)
        random.shuffle(self.args_house_inds)
        if self.sampler_mode == 'test':
            self.all_test_tasks = list(range(1000))
            self.max_tasks = 1000



    def reset_scene(self):
        self.env.reset(
            scene_name='Procedural',
            agentMode=self.env_args['agentMode'], agentControllerType=self.env_args['agentControllerType']
        )
    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.reset_scene()
        self.increment_scene_index()

        # self.controller.step(action="DestroyHouse", raise_for_failure=True)
        self.house_entry = self.house_dataset[self.house_index]
        self.house = pickle.loads(self.house_entry["house"])
        #TODO this probably needs to go on env side
        self.env.controller.reset()
        if platform.system() == "Darwin": #TODO remove
            print('The house is ', self.house_index)

        self.env.controller.step(action="CreateHouse", house=self.house,raise_for_failure=True)
        self.env.controller.step("ResetObjectFilter") # should we do after each reset?
        #TODO maybe use this after that for speed up
        # controller.step('SetObjectFilter', objectIds=['Mug|0.1|0.2|0.3|']) only for objects we care about

        #TODO dude this is ugly!
        pose = self.house["metadata"]["agent"].copy()
        event = self.env.controller.step(action="TeleportFull", **pose)
        if not event:
            get_logger().warning(f"Initial teleport failing in {self.house_index}.")
            return False #TODO this can mess FPS

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            # pose = self.house["metadata"]["agent"].copy()
            # event = self.env.controller.step(action="TeleportFull", **pose)
            # if not event:
            #     get_logger().warning(f"Initial teleport failing in {self.house_index}.")
            #     return False
            rp_event = self.env.controller.step(action="GetReachablePositions")
            if not rp_event or len(rp_event.metadata['actionReturn']) == 0:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in {self.house_index}"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.house_index] = reachable_positions
        else:
            reachable_positions = self.env.get_reachable_positions()
            if len(reachable_positions) == 0:
                print('no reachable positions' , self.house_index)
                return False
        return True

    def increment_scene_index(self):
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args_house_inds)
    @property
    def house_index(self) -> int:
        return self.args_house_inds[self.house_inds_index]


    def get_target_locations(self, forced_house_index=False, custom_index=0):
        #TODO Just for debugging
        if forced_house_index:
            scene_number = custom_index
        else:
            scene_number = self.house_index


        valid_objects = [o for o in self.env.controller.last_event.metadata['objects'] if o['objectType'] in self.objects]
        #TODO set the task to account for any object of type

        if len(valid_objects) == 0:
            print('FAILED TO FIND ANY VALID TASKS IN', scene_number)
            return None

        valid_object_types = [o['objectType'] for o in valid_objects]
        chosen_obj = random.choice(list(set(valid_object_types)))

        specific_object = random.choice([o for o in valid_objects if o['objectType'] == chosen_obj])
        source_obj = target_obj = specific_object
        agent_initial_pose = random.choice(self.reachable_positions_map[scene_number])

        return dict(
            source_obj=source_obj,
            target_obj=target_obj,
            agent_initial_pose=agent_initial_pose,
            scene_number=scene_number,
        )

    def next_task(
            self, force_advance_scene: bool = False
    ) :

        if self.env is None:
            self.env = self._create_environment()

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None




        if force_advance_scene or (
            self.resample_same_scene_freq > 0
            and self.episode_index % self.resample_same_scene_freq == 0
        ):

            while not self.increment_scene():
                print('scene', self.house_index, 'failed')
                pass

        data_point = self.get_target_locations()
        while data_point is None:
            print('no task was found, incrementing the scene')
            while not self.increment_scene():
                print('scene', self.house_index, 'failed')
                pass
            # self.increment_scene()
            data_point = self.get_target_locations()
        source_obj = data_point['source_obj']
        target_obj = data_point['target_obj']
        start_pose = data_point['agent_initial_pose']
        scene_number = data_point['scene_number']



        if random.random() < 0.5:
            self.env.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.env.controller.step(action="ResetMaterials", raise_for_failure=True)

        if random.random() < 0.5:
            self.env.controller.step(
                action="RandomizeLighting",
                brightness=(0.5, 1.5),
                randomizeColor=True,
                hue=(0, 1),
                saturation=(0.5, 1),
                synchronized=False
            )

        # NOTE: Set agent pose
        starting_pose = AgentPose(
            position=start_pose,
            rotation=dict(x=0,y=random.choice([i for i in range(0,360,90)]),z=0),
            horizon=random.choice(self.first_camera_horizon),
            standing=True,
        )
        event = self.env.controller.step(action="TeleportFull", **starting_pose)
        if not event:
            get_logger().warning(
                f"Teleport failing in {self.house_index} at {starting_pose}"
            )
        # else:
        #     get_logger().warning(
        #         f"Teleport succeeded in {self.house_index} at {starting_pose}"
        #     )

        self.episode_index += 1
        # self.max_tasks -= 1 TODO is none decrease when it's inference

        scene_name = self.house_index
        def convert_object_info(object_info_dict):
            object_info_dict['object_id'] = object_info_dict['objectId']
            object_info_dict['object_location'] = object_info_dict['position']
            object_info_dict['scene_name'] = f'ProcTHOR_{scene_number}'
            return object_info_dict
        init_object = convert_object_info(source_obj)
        goal_object = convert_object_info(target_obj)


        initial_object_info = self.env.get_object_by_id(init_object["object_id"])
        initial_agent_location = self.env.controller.last_event.metadata["agent"]
        initial_hand_state = self.env.get_absolute_hand_state()


        task_info = {
            'source_object_id': init_object['object_id'],
            'goal_object_id': goal_object['object_id'],
            'source_object_type': init_object['objectType'],
            'goal_object_type': goal_object['objectType'],
            "init_location": init_object,
            "goal_location": goal_object,
            'agent_initial_state': initial_agent_location,
            'initial_object_location':initial_object_info,
            'initial_hand_state': initial_hand_state,
            'episode_number': random.uniform(0, 10000),
            'scene_name':scene_name,
            'object_type': source_obj['objectType'],
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

            return min(self.max_tasks, len(self.all_test_tasks))


    @lazy_property
    def stretch_reachable_positions(self):
        return {}


    @lazy_property
    def cfg(self):
        with open("~/.hydra/config.yaml", "r") as f:
            cfg = OmegaConf.load(f.name)
        return cfg
