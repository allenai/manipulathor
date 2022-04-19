"""Task Samplers for the task of ArmPointNav"""
import json
import os
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
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from scripts.stretch_jupyter_helper import get_reachable_positions, transport_wrapper, AGENT_ROTATION_DEG
from utils.stretch_utils.stretch_visualizer import StretchBringObjImageVisualizer

class ProcTHORDiverseBringObjectTaskSampler(TaskSampler):

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
        env = StretchManipulaTHOREnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )

        return env

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

        # possible_initial_locations = (
        #     "datasets/apnd-dataset/valid_agent_initial_locations.json"
        # )
        #
        # with open(possible_initial_locations) as f:
        #     self.possible_agent_reachable_poses = json.load(f)

        # self.possible_agent_reachable_poses = {}# slowly load this one

        # self.query_image_dict = self.find_all_query_objects()
        # self.all_possible_points = {}
        # for scene in self.scenes:
        #     for object in self.objects:
        #         # valid_position_adr = "datasets/apnd-dataset/pruned_object_positions/pruned_v3_valid_{}_positions_in_{}.json".format(
        #         #     object, scene
        #         # )
        #         # should we generate kitchens again?
        #         if scene in KITCHEN_TRAIN + KITCHEN_TEST + KITCHEN_VAL:
        #             scene = scene + '_physics'
        #         valid_position_adr = "datasets/apnd-dataset/valid_object_positions/valid_{}_positions_in_{}.json".format(
        #             object, scene
        #         )
        #         if os.path.exists(valid_position_adr):
        #             with open(valid_position_adr) as f:
        #                 data_points = json.load(f)
        #                 assert len(data_points) == 1
        #                 only_key = [k for k in data_points][0]
        #                 data_points = data_points[only_key]
        #         else:
        #             continue
        #
        #         # if this is too big I can live with not having it and randomly sample each time
        #         all_locations_matrix = torch.tensor([[d['object_location']['x'], d['object_location']['y'], d['object_location']['z']] for d in data_points])
        #         self.all_possible_points[(scene, object)] = dict(
        #             data_point_dict=data_points,
        #             data_point_matrix=all_locations_matrix
        #         )
        #
        #
        # scene_names = set([x[0] for x in self.all_possible_points.keys()])
        #
        #
        #
        # if len(set(scene_names)) < len(self.scenes):
        #     print("Not all scenes appear")
        #     print([s for s in self.scenes if s not in scene_names])

        #

        self.house_dataset = datasets.load_dataset("allenai/houses", use_auth_token=True)

        RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = (
            -1
        )  # Should be > 0 if `ADVANCE_SCENE_ROLLOUT_PERIOD` is `None`
        RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 100
        self.resample_same_scene_freq = RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE
        self.episode_index = 0
        self.house_inds_index = 0
        self.reachable_positions_map = {}
        self.house_dataset = self.house_dataset['train'] #TODO separately for test and val
        self.args_house_inds = [i for i in range(len(self.house_dataset))]
        random.shuffle(self.args_house_inds)

        # len_all_data_points = [len(v['data_point_dict']) for v in self.all_possible_points.values()]
        # print(
        #     "Len dataset",
        #     sum(len_all_data_points),
        # )

        # if self.sampler_mode == "test":
        #     self.sampler_index = 0
        #     self.all_test_tasks = []
        #
        #     #TODO implement this
        #     if True:
        #         self.all_test_tasks = [i for i in range(1000)]
        #     else:
        #         ForkedPdb().set_trace()
        #
        #         for scene in self.scenes:
        #             for from_obj in self.objects:
        #                 for to_obj in self.objects:
        #                     if from_obj == to_obj:
        #                         continue
        #
        #                     with open(f'datasets/apnd-dataset/bring_object_deterministic_tasks/tasks_obj_{from_obj}_to_{to_obj}_scene_{scene}.json') as f:
        #                         tasks = json.load(f)['tasks']
        #
        #                     if from_obj in small_objects or to_obj in small_objects:
        #                         NUM_NEEDED = 1
        #                     else:
        #                         NUM_NEEDED = 2
        #
        #                     tasks = tasks[:NUM_NEEDED]
        #
        #                     self.all_test_tasks += tasks
        #     random.shuffle(self.all_test_tasks)
        #     self.max_tasks = self.reset_tasks = len(self.all_test_tasks)


    def reset_scene(self):
        self.env.reset( #TODO are these the correct values?
            scene_name='Procedural',
            agentMode="stretch", agentControllerType="mid-level"
        )
    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.reset_scene()
        self.increment_scene_index()

        # self.controller.step(action="DestroyHouse", raise_for_failure=True)
        # self.env.controller.reset() #TODO kiana removed this
        self.house_entry = self.house_dataset[self.house_index]
        self.house = pickle.loads(self.house_entry["house"])
        #TODO this probably needs to go on env side
        self.env.controller.reset()
        self.env.controller.step(action="CreateHouse", house=self.house,raise_for_failure=True)
        self.env.controller.step("ResetObjectFilter") #TODO should we do after each reset?

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            pose = self.house["metadata"]["agent"].copy()
            event = self.env.controller.step(action="TeleportFull", **pose)
            if not event:
                get_logger().warning(f"Initial teleport failing in {self.house_index}.")
                return False
            rp_event = self.env.controller.step(action="GetReachablePositions")
            if not rp_event:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in {self.house_index}"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.house_index] = reachable_positions
        return True

    def increment_scene_index(self):
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args_house_inds)
    @property
    def house_index(self) -> int:
        return self.args_house_inds[self.house_inds_index]
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

            while not self.increment_scene(): #TODO why?
                pass



        #TODO choose target objects
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                # target_object_type, target_object_ids = self.sample_target_object_ids()

                #TODO do we like the other approach?
                # valid_objects = [k for k in self.env.ob() if k['objectType'] in self.objects]

                valid_objects = [k for k in self.env.controller.last_event.metadata['objects'] if k['pickupable']]

                if len(valid_objects) < 2:
                    print('Not enough pickupable objects', len(valid_objects), 'room', self.house_index)
                    raise ValueError()

                source_obj, target_obj = random.sample(valid_objects, 2) #TODO this might be invalid
                source_object_type = source_obj['objectId'], source_obj['objectType']
                if not (source_obj['pickupable']):
                    print(f'Source object not pickupable {source_object_type}')
                    raise ValueError()

                break
            except ValueError:
                while not self.increment_scene():
                    pass

        if False and random.random() < self.cfg.procthor.p_randomize_materials: #TODO
            self.env.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.env.controller.step(action="ResetMaterials", raise_for_failure=True)

        # self.env.controller.step( #TODO what is this?
        #     action="SetObjectFilter",
        #     objectIds=target_object_ids,
        #     raise_for_failure=True,
        # )

        #TODO choose agent pose
        # NOTE: Set agent pose
        start_pose = random.choice(self.reachable_positions_map[self.house_index])
        # standing = (
        #     {}
        #     if self.args.controller_args["agentMode"] == "locobot"
        #     else {"standing": True}
        # )
        starting_pose = AgentPose(
            position=start_pose,
            rotation=Vector3(x=0, y=random.choice([i for i in range(0,360,AGENT_ROTATION_DEG)]), z=0),
            horizon=0,
            standing=True,
        )
        event = self.env.controller.step(action="TeleportFull", **starting_pose)
        if not event:
            get_logger().warning(
                f"Teleport failing in {self.house_index} at {starting_pose}"
            )

        self.episode_index += 1
        # self.max_tasks -= 1 #TODO is none

        scene_name = self.house_index
        def convert_object_info(object_info_dict):
            object_info_dict['object_id'] = object_info_dict['objectId']
            object_info_dict['object_location'] = object_info_dict['position']
            object_info_dict['scene_name'] = 'ProcTHOR'
            return object_info_dict
        init_object = convert_object_info(source_obj)
        goal_object = convert_object_info(target_obj)
        # agent_state = data_point["initial_agent_pose"]
        # agent_state = {
        #         "name": "agent",
        #         "position": starting_pose['position'],
        #         "rotation": starting_pose['position']['rotation'],
        #         "horizon": 0,
        #         "standing": True,
        # }



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
            'scene_name':scene_name
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
        # with open('datasets/apnd-dataset/stretch_init_location.json') as f:
        #
        #     scene_name_to_locations_dict = json.load(f)
        # all_keys = [k for k in scene_name_to_locations_dict.keys()]
        # for k in all_keys:
        #     if '_physics' in k:
        #         scene_name_to_locations_dict[k.replace('_physics', '')] = scene_name_to_locations_dict[k]
        # return scene_name_to_locations_dict

    # def get_stretch_reachable_positions(self, scene_name):
    #     if scene_name in self.stretch_reachable_positions:
    #         return self.stretch_reachable_positions[scene_name]
    #     else:
    #         reachable_positions = get_reachable_positions_procthor(self.env.controller)
    #         self.stretch_reachable_positions[scene_name] = reachable_positions # TODO will this overflow?
    #         return reachable_positions
    @lazy_property
    def cfg(self):
        with open("~/.hydra/config.yaml", "r") as f:
            cfg = OmegaConf.load(f.name)
        return cfg
    def get_source_target_indices(self): #TODO need to implement this one
        if self.sampler_mode == "train" or True: #TODO remove or TRUE

            ForkedPdb().set_trace()
            if False and random.random() < self.cfg.procthor.p_randomize_materials: #TODO
                self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
            else:
                self.controller.step(action="ResetMaterials", raise_for_failure=True)
            chosen_scene = random.choice(self.house_dataset)
            all_scenes = [s for (s,o) in self.all_possible_points.keys()]


            #randomly choose a scene
            chosen_scene = random.choice(all_scenes)
            all_objects = [o for (s, o) in self.all_possible_points.keys() if s == chosen_scene]

            #randomly choosing initial and goal objects, the whole following needs to be changed if working with static objects
            init_obj, goal_obj = random.sample(all_objects, 2)
            #randomly choosing an initial location for first object
            init_object_location = random.choice(self.all_possible_points[(chosen_scene, init_obj)]['data_point_dict'])
            initial_location = torch.tensor([init_object_location['object_location']['x'], init_object_location['object_location']['y'], init_object_location['object_location']['z']])

            #calulcate distance of initial object location to all the possible target location
            all_goal_object_locations = self.all_possible_points[(chosen_scene, goal_obj)]['data_point_matrix']
            all_distances = (all_goal_object_locations - initial_location).norm(2, dim=-1)

            #randomly choosing a target location which is far enough from the initial location therefore chances of collisions and failures are low
            valid_goal_indices = torch.nonzero(all_distances > 1.0)
            if len(valid_goal_indices) == 0:
                print('No far goal was found', chosen_scene, init_obj, goal_obj, 'max distance', all_distances.max())
                valid_goal_indices = torch.nonzero(all_distances > 0)
            chosen_goal_instance = random.choice(valid_goal_indices)
            goal_object_location = self.all_possible_points[(chosen_scene, goal_obj)]['data_point_dict'][chosen_goal_instance]

            data_point = {}
            data_point['scene_name'] = chosen_scene
            data_point['init_object'] = init_object_location
            data_point['goal_object'] = goal_object_location
            # data_point["initial_agent_pose"] = initial_agent_pose

            #TODO this needs to be fixed for test
            reachable_positions = self.get_stretch_reachable_positions(data_point['scene_name'])
            agent_pose = random.choice(reachable_positions)



            data_point['initial_agent_pose'] = {
                "name": "agent",
                "position": dict(x=agent_pose['x'], y=agent_pose['y'], z=agent_pose['z']),
                "rotation": dict(x=0, y=agent_pose['rotation'], z=0),
                "cameraHorizon": agent_pose['horizon'],
                "isStanding": True,
            }

        else:

            task = self.all_test_tasks[self.sampler_index]
            self.sampler_index += 1

            data_point = task




        return data_point
