"""Task Samplers for the task of ArmPointNav"""
from ast import Continue
from datetime import datetime
import json
import os, platform
import random
from typing import Optional, List, Union, Dict, Any

import gym
import torch
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler, Task
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger

from torch.distributions.utils import lazy_property
from allenact.utils.cache_utils import DynamicDistanceCache
from collections import Counter
from utils.procthor_utils.procthor_types import AgentPose, Vector3

from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import ROBOTHOR_TRAIN, KITCHEN_TRAIN, KITCHEN_TEST, KITCHEN_VAL
from utils.manipulathor_data_loader_utils import get_random_query_image, get_random_query_feature_from_img_adr
from scripts.stretch_jupyter_helper import get_reachable_positions, transport_wrapper, is_arm_stowed
from utils.stretch_utils.stretch_object_nav_tasks import ObjectNavTask
from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer

from utils.stretch_utils.real_stretch_environment import StretchRealEnvironment
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment


class AllRoomsObjectNavTaskSampler(TaskSampler):

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


    def _create_environment(self, **kwargs) -> IThorEnvironment:
        env = self.environment_type(
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
        process_ind: int,
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

        self.scenes = scenes
        self.grid_size = 0.25
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.objects = objects
        self.environment_type = env_args['environment_type']
        del env_args['environment_type']
        self.env_args = env_args

        self.distance_type = "l2"
        self.distance_cache = DynamicDistanceCache(rounding=1)

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

        self.success_distance = 1.0

        possible_initial_locations = (
            "datasets/apnd-dataset/valid_agent_initial_locations.json"
        )

        with open(possible_initial_locations) as f:
            self.possible_agent_reachable_poses = json.load(f)

        # self.query_image_dict = self.find_all_query_objects()
        self.all_possible_points = {}
        for scene in self.scenes:
            for object in self.objects:

                if scene in KITCHEN_TRAIN + KITCHEN_TEST + KITCHEN_VAL:
                    scene = scene + '_physics'
                valid_position_adr = "datasets/apnd-dataset/valid_object_positions/valid_{}_positions_in_{}.json".format(
                    object, scene
                )
                if os.path.exists(valid_position_adr):
                    with open(valid_position_adr) as f:
                        data_points = json.load(f)
                        assert len(data_points) == 1
                        only_key = [k for k in data_points][0]
                        data_points = data_points[only_key]
                else:
                    continue

                # if this is too big I can live with not having it and randomly sample each time
                all_locations_matrix = torch.tensor([[d['object_location']['x'], d['object_location']['y'], d['object_location']['z']] for d in data_points])
                self.all_possible_points[(scene, object)] = dict(
                    data_point_dict=data_points,
                    data_point_matrix=all_locations_matrix
                )

        scene_names = set([x[0] for x in self.all_possible_points.keys()])

        if len(set(scene_names)) < len(self.scenes):
            print("Not all scenes appear")
            print([s for s in self.scenes if s not in scene_names])

        len_all_data_points = [len(v['data_point_dict']) for v in self.all_possible_points.values()]
        print(
            "Len dataset",
            sum(len_all_data_points),
        )

        if self.sampler_mode == "test":
            self.sampler_index = 0
            self.all_test_tasks = []

            with open('datasets/apnd-dataset/deterministic_stretch_init_location.json') as f: #TODO how many teleport fails?
                deterministic_locations = json.load(f)
            for scene in self.scenes:
                objects_in_room = [o for (s,o) in self.all_possible_points.keys() if s == scene]
                if scene in deterministic_locations:
                    list_of_possible_locations = deterministic_locations[scene]
                else:
                    list_of_possible_locations = deterministic_locations[scene + '_physics']
                # print('scene, ', len(list_of_possible_locations)) #
                task_number = -1
                for target_obj in objects_in_room:

                    task_number += 1
                    agent_pose = list_of_possible_locations[task_number % len(list_of_possible_locations)]

                    task = {
                        "scene_name": scene,
                        # "target_obj": target_obj,
                        "object_type": target_obj,
                        "agent_initial_location": agent_pose,
                    }
                    self.all_test_tasks.append(task)
            random.shuffle(self.all_test_tasks)
            self.max_tasks = self.reset_tasks = len(self.all_test_tasks)

    def reset_scene(self, scene_name):
        self.env.reset(
            scene_name=scene_name, agentMode="stretch", agentControllerType="mid-level"
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

        data_point = self.get_source_target_indices()

        scene_name = data_point["scene_name"]

        agent_state = data_point["initial_agent_pose"]

        self.reset_scene(scene_name)

        this_controller = self.env

        if self.sampler_mode == 'train':
            init_object = data_point['init_object']
            # just because name of objects are changed
            def convert_name(env, object_info):
                prev_object_id = object_info['object_id']
                if env.get_object_by_id(prev_object_id) is None:
                    object_type = prev_object_id.split('|') [0]
                    objects_of_type = env.controller.last_event.objects_by_type(object_type)
                    if len(objects_of_type) > 1:
                        print('MULTIPLE ANSWERS', object_type, env.controller.last_event.metadata['sceneName'])
                    target_object = random.choice(objects_of_type)
                    object_info['object_id'] = target_object['objectId']
                object_info['object_type'] = object_info['object_id'].split('|') [0]
                return object_info
            init_object = convert_name(self.env, init_object)
            # goal_object = convert_name(self.env, goal_object)

            assert init_object["scene_name"] == scene_name
            # assert init_object['object_id'] != goal_object['object_id']

            def put_object_in_location(location_point):

                object_id = location_point['object_id']
                location = location_point['object_location']
                event, _ = transport_wrapper(this_controller,object_id,location,)
                return event

            event_transport_init_obj = put_object_in_location(init_object)
            # event_transport_goal_obj = put_object_in_location(goal_object)

            # if not event_transport_init_obj.metadata['lastActionSuccess']:
                # print('scene', scene_name, 'init', init_object['object_id'])
                # print('ERROR: one of transfers fail', 'init', event_transport_init_obj.metadata['errorMessage'])

            # if random.random() < 0.8:
            #     # self.env.controller.step(action="RandomizeMaterials", raise_for_failure=True)
            #     self.env.controller.step(action="RandomizeLighting", synchronized=True, raise_for_failure=True)
            # else:
                # self.env.controller.step(action="ResetMaterials", raise_for_failure=True)

        else:
            #Object is already at the location it should be
            target_obj_type = data_point['target_obj_type']

            def get_full_object_info(env, object_type):
                object_info = env.get_object_by_type(object_type)
                if len(object_info) > 1:
                    print('multiple objects of this type!!!!', scene_name, object_type)
                object_info = object_info[0]
                object_info['object_id'] = object_info['objectId']
                object_info['object_type'] = object_type
                object_info['object_location'] = object_info['position']
                object_info['scene_name'] = scene_name
                return object_info

            init_object = get_full_object_info(self.env, target_obj_type)

        agent_state["cameraHorizon"] = self.env_args['horizon_init'] # 0 for stretch, 20 for other manipulathor agent
        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )

        # if not event.metadata['lastActionSuccess']:
        #     print('ERROR: Teleport failed')


        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), StretchObjNavImageVisualizer)
        ]

        initial_agent_location = self.env.controller.last_event.metadata["agent"]

        task_info = {
            'target_object_ids': [init_object['object_id']],
            'object_type': init_object['object_type'],
            'starting_pose': initial_agent_location,
            'mode': self.env_args['agentMode'],
            'house_name': scene_name,
            'scene_name': scene_name,
            'mirrored': False, #self.env_args['allow_flipping'] and random.random() > 0.5, # not including for non-procthor
            'success_distance': self.success_distance
        }


        if len(should_visualize_goal_start) > 0:
            task_info["visualization_source"] = init_object

        self._last_sampled_task = self.TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_config=self.rewards_config,
            distance_type=self.distance_type,
            distance_cache=self.distance_cache,
            additional_visualize=False # not implemented for non-procthor
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
        with open('datasets/apnd-dataset/stretch_init_location.json') as f:

            scene_name_to_locations_dict = json.load(f)
        all_keys = [k for k in scene_name_to_locations_dict.keys()]
        for k in all_keys:
            if '_physics' in k:
                scene_name_to_locations_dict[k.replace('_physics', '')] = scene_name_to_locations_dict[k]
        return scene_name_to_locations_dict

    def get_source_target_indices(self):
        if self.sampler_mode == "train":
            all_scenes = [s for (s,o) in self.all_possible_points.keys()]
            

            #randomly choose a scene
            chosen_scene = random.choice(all_scenes)
            all_objects = [o for (s, o) in self.all_possible_points.keys() if s == chosen_scene]

            #randomly choosing initial and goal objects, the whole following needs to be changed if working with static objects
            init_obj, goal_obj = random.sample(all_objects, 2)
            #randomly choosing an initial location for first object
            init_object_location = random.choice(self.all_possible_points[(chosen_scene, init_obj)]['data_point_dict'])

            data_point = {}
            data_point['scene_name'] = chosen_scene
            data_point['init_object'] = init_object_location

            reachable_positions = self.stretch_reachable_positions[data_point['scene_name']] #TODO does it work for manipulathor agent as well?
            agent_pose = random.choice(reachable_positions)


            data_point['initial_agent_pose'] = {
                "name": "agent",
                "position": dict(x=agent_pose['x'], y=agent_pose['y'], z=agent_pose['z']),
                "rotation": dict(x=0, y=agent_pose['rotation'], z=0),
                "cameraHorizon": self.env_args['horizon_init'], #agent_pose['horizon'],
                "isStanding": True,
            }

        else:

            task = self.all_test_tasks[self.sampler_index]
            data_point = {}
            data_point['scene_name'] = task['scene_name']
            data_point['target_obj_type'] = (task['object_type'])

            agent_pose = task['agent_initial_location']

            data_point['initial_agent_pose'] = {
                "name": "agent",
                "position": dict(x=agent_pose['x'], y=agent_pose['y'], z=agent_pose['z']),
                "rotation": dict(x=0, y=agent_pose['rotation'], z=0),
                "cameraHorizon": self.env_args['horizon_init'], #agent_pose['horizon'],
                "isStanding": True,
            }

            self.sampler_index += 1



        return data_point
    

class RoboTHORObjectNavTaskSampler(TaskSampler):
    # Train only
    def __init__(
        self,
        scenes: List[str],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        # process_ind: int,
        objects: List[str],
        task_type: type,
        distance_type: Optional[str] = "l2",
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
        self.env: Optional[StretchManipulaTHOREnvironment] = None
        del env_args['environment_type']
        self.env_args = env_args
        self.scenes = scenes
        self.grid_size = 0.25
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        
        self.target_object_types_set = set(objects)
        self.obj_type_counter = Counter(
            {obj_type: 0 for obj_type in objects}
        )
        self.distance_type = distance_type
        self.distance_cache = DynamicDistanceCache(rounding=1)
        self.episode_index = 0
        self.success_distance = 1.0
        # self.process_ind = kwargs['process_ind']

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

        self.reachable_positions_map = dict()
        self.objects_in_scene_map = dict()
        self.p_greedy_target_object = 0.8


        random.shuffle(self.scenes)
        if self.sampler_mode == 'test':
            self.all_test_tasks = list(range(1000))
            self.max_tasks = 1000


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

    @property
    def target_objects_in_scene(self) -> Dict[str, List[str]]:
        """Return a map from the object type to the objectIds in the scene."""
        if self.env.scene_name in self.objects_in_scene_map:
            return self.objects_in_scene_map[self.env.scene_name]

        event = self.env.controller.step(action="ResetObjectFilter", raise_for_failure=True)
        all_objects = event.metadata["objects"]
        out = {}
        for obj in all_objects:
            if obj["objectType"] in self.target_object_types_set:
                if obj["objectType"] not in out:
                    out[obj["objectType"]] = []
                out[obj["objectType"]].append(obj["objectId"])
        self.objects_in_scene_map[self.env.scene_name] = out
        return out
    

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.env.scene_name] 
    
    def sample_target_object_ids(self,forced_type=None):
        """Sample target objects.
        Objects returned will all be of the same objectType. Only considers visible
        objects in the house.
        """

        if random.random() < self.p_greedy_target_object:
            for obj_type, count in reversed(self.obj_type_counter.most_common()):
                instances_of_type = self.target_objects_in_scene.get(obj_type, [])


                # NOTE: object type doesn't appear in the scene.
                if not instances_of_type:
                    continue

                visible_ids = []
                for object_id in instances_of_type:
                    # if self.is_object_visible(object_id=object_id):
                    visible_ids.append(object_id)

                if visible_ids:
                    self.obj_type_counter[obj_type] += 1
                    return obj_type, visible_ids
        else:
            candidates = dict()
            for obj_type, object_ids in self.target_objects_in_scene.items():
                visible_ids = []
                for object_id in object_ids:
                    # if self.is_object_visible(object_id=object_id):
                    visible_ids.append(object_id)

                if visible_ids:
                    candidates[obj_type] = visible_ids

            if candidates:
                return random.choice(list(candidates.items()))

        raise ValueError(f"No target objects in house {self.scene_id}.")
    
    def increment_scene(self) -> bool:
        this_scene = random.choice(self.scenes)
        self.env.reset(scene_name='Procedural',scene=this_scene)
        if platform.system() == "Darwin":
            print('The house is ', this_scene)
        
        if self.env.scene_name not in self.reachable_positions_map:
            rp_event = self.env.controller.step(action="GetReachablePositions")
            if not rp_event:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in {self.env.scene_name}"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.env.scene_name] = reachable_positions

        # assert is_arm_stowed(self.env.controller)
        attempts=0
        while not is_arm_stowed(self.env.controller):
            self.env.controller.step(dict(action='MoveArm', position=dict(x=0, y=0.1, z=0)))
            self.env.controller.step({"action":'RotateWristRelative',"yaw":90})
            attempts+=1
            if attempts>6:
                get_logger().error(
                    f"Arm stowing failed in {self.env.scene_name}" # rare but possible
                )
                return False
        return True

        
    def next_task(self, force_advance_scene: bool = False):
        if self.env is None:
            self.env = self._create_environment()

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None
        
        while not self.increment_scene():
            pass
        
        target_object_type, target_object_ids = self.sample_target_object_ids()
        # self.env.controller.step(
        #     action="SetObjectFilter",
        #     objectIds=target_object_ids,
        #     raise_for_failure=True,
        # )

        event = None
        attempts = 0
        while not event:
            attempts+=1
            starting_pose = AgentPose(
                position=random.choice(self.reachable_positions),
                rotation=Vector3(x=0, y=random.choice([i for i in range(0,360,30)]), z=0),
                horizon=0,
            )
            if self.env_args['agentMode'] != 'locobot':
                starting_pose['standing']=True
                starting_pose['horizon'] = self.env_args['horizon_init'] + random.gauss(0,5)
            event = self.env.controller.step(action="TeleportFull", **starting_pose)
            if attempts > 10:
                get_logger().error(f"Teleport failed {attempts-1} times in house {self.house_index} - something may be wrong")
            
            
        self.episode_index += 1
        # self.max_tasks -= 1
        
        self._last_sampled_task = self.TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            max_steps=self.max_steps,
            reward_config=self.rewards_config,
            distance_type=self.distance_type,
            distance_cache=self.distance_cache,
            visualizers=self.visualizers,
            # visualize=True,
            task_info={
                "mode": self.sampler_mode, #self.env_args['agentMode'],
                "process_ind": 0,#self.process_ind,
                # "scene_name": self.env_args['scene'],
                "house_name": str(self.env.scene_name),
                "rooms": 1,#self.house["rooms"],
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": False,#self.env_args['allow_flipping'] and random.random() > 0.5,
                'success_distance': self.success_distance
            },
        )
        return self._last_sampled_task


class RealStretchAllRoomsObjectNavTaskSampler(AllRoomsObjectNavTaskSampler):

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

        # This can be done more elegantly from datasets/objects/robothor_habitat2022.yaml and exclude toilet
        self.possible_real_objects = [
            {'object_id': 'AlarmClock|1|1|1', 'object_type':"AlarmClock"},
            {'object_id': 'Apple|1|1|1', 'object_type':"Apple"},
            {'object_id': 'BaseballBat|1|1|1', 'object_type':"BaseballBat"},
            {'object_id': 'BasketBall|1|1|1', 'object_type':"BasketBall"},
            {'object_id': 'Bed|1|1|1', 'object_type':"Bed"},
            {'object_id': 'Bowl|1|1|1', 'object_type':"Bowl"},
            {'object_id': 'Chair|1|1|1', 'object_type':"Chair"},
            {'object_id': 'GarbageCan|1|1|1', 'object_type':"GarbageCan"},
            {'object_id': 'HousePlant|1|1|1', 'object_type':"HousePlant"},
            {'object_id': 'Laptop|1|1|1', 'object_type':"Laptop"},
            {'object_id': 'Mug|1|1|1', 'object_type':"Mug"},
            {'object_id': 'Sofa|1|1|1', 'object_type':"Sofa"},
            {'object_id': 'SprayBottle|1|1|1', 'object_type':"SprayBottle"},
            {'object_id': 'Television|1|1|1', 'object_type':"Television"},
            {'object_id': 'Vase|1|1|1', 'object_type':"Vase"}
        ]

        self.preset_easyish_tasks = [
            # {'object_id': 'Bed|1|1|1', 'object_type':"Bed"},
            # {'object_id': 'Sofa|1|1|1', 'object_type':"Sofa"},
            # {'object_id': 'Apple|1|1|1', 'object_type':"Apple"},
            # {'object_id': 'BaseballBat|1|1|1', 'object_type':"BaseballBat"},
            # {'object_id': 'BasketBall|1|1|1', 'object_type':"BasketBall"},
            # {'object_id': 'Bowl|1|1|1', 'object_type':"Bowl"},
            # {'object_id': 'Chair|1|1|1', 'object_type':"Chair"},
            {'object_id': 'HousePlant|1|1|1', 'object_type':"HousePlant"},
            {'object_id': 'Mug|1|1|1', 'object_type':"Mug"},
            {'object_id': 'SprayBottle|1|1|1', 'object_type':"SprayBottle"},
            {'object_id': 'Television|1|1|1', 'object_type':"Television"},
            {'object_id': 'Vase|1|1|1', 'object_type':"Vase"}
        ]

        self.real_object_index = 0

        if self.sampler_mode == "test":
            self.max_tasks = self.reset_tasks = 200

    def next_task(
            self, force_advance_scene: bool = False
    ) -> Optional[ObjectNavTask]:

        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        if self.env is None:
            self.env = self._create_environment()
        self.env.reset(scene_name='RealRobothor')
        
        # skip_object = True
        # while skip_object:
        #     target_object = random.choice(self.possible_real_objects)
        #     print('I am now seeking a', target_object['object_type'], '. Accept by setting skip_object=False')
        #     ForkedPdb().set_trace()
        
        target_object = self.preset_easyish_tasks[self.real_object_index]
        print('I am now seeking a', target_object['object_type'], '. Continue when ready.')
        ForkedPdb().set_trace()

        # do this to reset the camera/sensors after moving the robot/resetting environment
        self.env.step({"action": "Done"})
        
        task_start = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.TASK_TYPE.__name__))
        task_info = {
            'target_object_ids': [target_object['object_id']],
            'object_type': target_object['object_type'],
            'starting_pose': {},
            'mode': self.env_args['agentMode'],
            'house_name': 'RealRobothor',
            "id": f"realRobothor__{target_object['object_type']}__{task_start}",
            # 'scene_name': 'RealRobothor',
            'mirrored': False # yeah no
        }

        self._last_sampled_task = self.TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_config=self.rewards_config,
            distance_type="real_world",
            distance_cache=self.distance_cache,
            additional_visualize=False # not implemented for non-procthor
        )
        self.real_object_index = self.real_object_index + 1

        return self._last_sampled_task