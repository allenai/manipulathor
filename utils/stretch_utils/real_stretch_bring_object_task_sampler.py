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


class RealStretchDiverseBringObjectTaskSampler(DiverseBringObjectTaskSampler):

    def _create_environment(self, **kwargs) -> ManipulaTHOREnvironment:
        env = StretchRealEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )
        return env

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        # TODO this needs to be changed later
        # if self.sampler_mode == "test":
        #     possible_initial_locations = (
        #         "datasets/apnd-dataset/deterministic_valid_agent_initial_locations.json"
        #     )
        # with open(possible_initial_locations) as f:
        #     self.possible_agent_reachable_poses = json.load(f)

        self.query_image_dict = self.find_all_query_objects()
        self.all_possible_points = {}

        if self.sampler_mode == "test":
            self.max_tasks = self.reset_tasks = 200


    def find_all_query_objects(self):
        IMAGE_DIR = 'datasets/apnd-dataset/query_images/'
        all_object_types = [f.split('/')[-1] for f in glob.glob(os.path.join(IMAGE_DIR, '*'))]
        all_possible_images = {object_type: [f for f in glob.glob(os.path.join(IMAGE_DIR, object_type, '*.png'))] for object_type in all_object_types}
        return all_possible_images

    def next_task(
            self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        if self.env is None:
            self.env = self._create_environment()
        self.env.reset(scene_name='KianaRoom')
        init_object = {'object_id': 'Apple|1|1|1'}
        goal_object = {'object_id': 'Mug|1|1|1'}

        def load_and_resize(img_name):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            with open(img_name, 'rb') as fp:
                image = Image.open(fp).convert('RGB')
            return transform(image)
        def get_random_query_image(object_id):
            object_category = object_id.split('|')[0]
            # object_type = object_category[0].lower() + object_category[1:]
            chosen_image_adr = random.choice(self.query_image_dict[object_category])
            image = load_and_resize(chosen_image_adr)
            return image

        source_object_query = get_random_query_image(init_object['object_id'])
        goal_object_query = get_random_query_image(goal_object['object_id'])

        task_info = {
            'source_object_id': init_object['object_id'],
            'goal_object_id': goal_object['object_id'],
            "init_location": {},
            "goal_location": {},
            'agent_initial_state': {},
            'initial_object_location':{},
            'initial_hand_state': {},
            'source_object_query':source_object_query,
            'goal_object_query': goal_object_query,
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