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
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import BringObjectTask, WPickUpBringObjectTask, WPickUPExploreBringObjectTask, NoPickUPExploreBringObjectTask
from ithor_arm.ithor_arm_constants import transport_wrapper
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_tasks import (
    AbstractPickUpDropOffTask,
)
from ithor_arm.ithor_arm_viz import LoggerVisualizer, BringObjImageVisualizer
from manipulathor_baselines.stretch_bring_object_baselines.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.stretch_jupyter_helper import get_reachable_positions
from utils.manipulathor_data_loader_utils import get_random_query_image


class StretchDiverseBringObjectTaskSampler(DiverseBringObjectTaskSampler):
    def _create_environment(self, **kwargs) -> StretchManipulaTHOREnvironment:
        env = StretchManipulaTHOREnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )

        return env
    def get_source_target_indices(self):
        data_point = super().get_source_target_indices()


        # data_point['initial_agent_pose']['position']['y'] += 0.02 #TODO is this really a good idea? or just a quick hack?
        # return data_point


        #TODO this is a quick hack we need to find a better solution DEFNITELY won't work for test
        self.env.reset(
            scene_name=data_point['scene_name'], agentMode="arm", agentControllerType="mid-level"
        ) #TODO this is happening twice!!1
        reachable_positions = get_reachable_positions( self.env.controller)
        chosen_position = random.choice(reachable_positions)
        chosen_rotation = {'x':0, 'y':random.choice([i * 30 for i in range(120)]), 'z':0}
        # reachable_positions = self.env.reachable_points_with_rotations_and_horizons() #TODO maybe this function is not working

        random_point = random.choice(reachable_positions)
        data_point['initial_agent_pose'] = {
            "name": "agent",
            "rotation": chosen_rotation,
            "position": chosen_position,
            "cameraHorizon": data_point['initial_agent_pose']['cameraHorizon'],
            "isStanding": True,
        }

        return data_point
