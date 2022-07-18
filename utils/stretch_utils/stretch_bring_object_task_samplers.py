"""Task Samplers for the task of ArmPointNav"""
import json
import os
import random
from typing import Optional, List, Union, Dict, Any

import gym
import torch
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler, Task
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from torch.distributions.utils import lazy_property

from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.dataset_generation.find_categories_to_use import ROBOTHOR_TRAIN, KITCHEN_TRAIN, KITCHEN_TEST, KITCHEN_VAL
from utils.manipulathor_data_loader_utils import get_random_query_image, get_random_query_feature_from_img_adr
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from scripts.stretch_jupyter_helper import get_reachable_positions, transport_wrapper
from utils.stretch_utils.stretch_visualizer import StretchBringObjImageVisualizer


class StretchDiverseBringObjectTaskSampler(TaskSampler):

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
        self.first_camera_horizon = kwargs['first_camera_horizon']
        del env_args['environment_type']
        assert issubclass(self.environment_type, StretchManipulaTHOREnvironment) # check that the env is a child of stretch?
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

        possible_initial_locations = (
            "datasets/apnd-dataset/valid_agent_initial_locations.json"
        )

        with open(possible_initial_locations) as f:
            self.possible_agent_reachable_poses = json.load(f)

        # self.query_image_dict = self.find_all_query_objects()
        self.all_possible_points = {}
        for scene in self.scenes:
            for object in self.objects:
                # valid_position_adr = "datasets/apnd-dataset/pruned_object_positions/pruned_v3_valid_{}_positions_in_{}.json".format(
                #     object, scene
                # )
                # should we generate kitchens again?
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

        # TODO just stats
        # for obj in self.objects:
        #     scene_names_for_obj = set([x[0] for x in self.all_possible_points.keys() if x[1] == obj])
        #     print(obj, ':', len(scene_names_for_obj))




        len_all_data_points = [len(v['data_point_dict']) for v in self.all_possible_points.values()]
        print(
            "Len dataset",
            sum(len_all_data_points),
        )

        if self.sampler_mode == "test":
            self.sampler_index = 0
            self.all_test_tasks = []

            # we need to fix this later
            # small_objects = ['Spatula', 'Egg']
            small_objects = []


            #TODO implement this
            if True:
                self.all_test_tasks = [i for i in range(1000)]
            else:
                raise Exception('not implemented')

                for scene in self.scenes:
                    for from_obj in self.objects:
                        for to_obj in self.objects:
                            if from_obj == to_obj:
                                continue

                            with open(f'datasets/apnd-dataset/bring_object_deterministic_tasks/tasks_obj_{from_obj}_to_{to_obj}_scene_{scene}.json') as f:
                                tasks = json.load(f)['tasks']

                            if from_obj in small_objects or to_obj in small_objects:
                                NUM_NEEDED = 1
                            else:
                                NUM_NEEDED = 2

                            tasks = tasks[:NUM_NEEDED]

                            self.all_test_tasks += tasks
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
        init_object = data_point['init_object']
        goal_object = data_point['goal_object']
        agent_state = data_point["initial_agent_pose"]

        self.reset_scene(scene_name)
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
            return object_info


        init_object = convert_name(self.env, init_object)
        goal_object = convert_name(self.env, goal_object)

        assert init_object["scene_name"] == goal_object["scene_name"] == scene_name
        assert init_object['object_id'] != goal_object['object_id']



        this_controller = self.env



        def put_object_in_location(location_point):

            object_id = location_point['object_id']
            location = location_point['object_location']
            event, _ = transport_wrapper(this_controller,object_id,location,)
            return event

        event_transport_init_obj = put_object_in_location(init_object)
        event_transport_goal_obj = put_object_in_location(goal_object)

        if not event_transport_goal_obj.metadata['lastActionSuccess'] or not event_transport_init_obj.metadata['lastActionSuccess']:
            print('scene', scene_name, 'init', init_object['object_id'], 'goal', goal_object['object_id'])
            print('ERROR: one of transfers fail', 'init', event_transport_init_obj.metadata['errorMessage'], 'goal', event_transport_goal_obj.metadata['errorMessage'])

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
                # horizon=agent_state["cameraHorizon"],
                horizon = self.first_camera_horizon,
            )
        )
        if not event.metadata['lastActionSuccess']:
            print('ERROR: Teleport failed') #TODO NOW if this fails the horizon thing wont work


        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), StretchBringObjImageVisualizer)
        ]

        initial_object_info = self.env.get_object_by_id(init_object["object_id"])
        initial_agent_location = self.env.controller.last_event.metadata["agent"]
        initial_hand_state = self.env.get_absolute_hand_state()

        # source_img_adr = get_random_query_image_file_name(scene_name,init_object['object_id'], self.query_image_dict)
        # goal_img_adr = get_random_query_image_file_name(scene_name,goal_object['object_id'], self.query_image_dict)

        # source_object_query, source_img_adr = get_random_query_image(scene_name,init_object['object_id'], self.query_image_dict)
        # goal_object_query, goal_img_adr = get_random_query_image(scene_name,goal_object['object_id'], self.query_image_dict)
        # source_object_query_feature = get_random_query_feature_from_img_adr(source_img_adr)
        # goal_object_query_feature = get_random_query_feature_from_img_adr(goal_img_adr)


        task_info = {
            'source_object_id': init_object['object_id'],
            'goal_object_id': goal_object['object_id'],
            "init_location": init_object,
            "goal_location": goal_object,
            'agent_initial_state': initial_agent_location,
            'initial_object_location':initial_object_info,
            'initial_hand_state': initial_hand_state,
            # 'source_object_query': source_object_query,
            # 'goal_object_query': goal_object_query,
            # 'source_object_query_feature': source_object_query_feature,
            # 'goal_object_query_feature': goal_object_query_feature,
            # 'source_object_query_file_name' : source_img_adr,
            # 'goal_object_query_file_name' : goal_img_adr,
            'episode_number': random.uniform(0, 10000),
            'scene_name':scene_name
        }


        if len(should_visualize_goal_start) > 0:
            task_info["visualization_source"] = init_object
            task_info["visualization_target"] = goal_object

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
        with open('datasets/apnd-dataset/stretch_init_location.json') as f:

            scene_name_to_locations_dict = json.load(f)
        all_keys = [k for k in scene_name_to_locations_dict.keys()]
        for k in all_keys:
            if '_physics' in k:
                scene_name_to_locations_dict[k.replace('_physics', '')] = scene_name_to_locations_dict[k]
        return scene_name_to_locations_dict

    def get_source_target_indices(self):
        if self.sampler_mode == "train" or True: #TODO remove or TRUE
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


            # selected_agent_init_loc = random.choice(
            #     self.possible_agent_reachable_poses[chosen_scene]
            # )
            # initial_agent_pose = {
            #     "name": "agent",
            #     "position": {
            #         "x": selected_agent_init_loc["x"],
            #         "y": selected_agent_init_loc["y"],
            #         "z": selected_agent_init_loc["z"],
            #     },
            #     "rotation": {
            #         "x": -0.0,
            #         "y": selected_agent_init_loc["rotation"],
            #         "z": 0.0,
            #     },
            #     "cameraHorizon": selected_agent_init_loc["horizon"],
            #     "isStanding": True,
            # }



            data_point = {}
            data_point['scene_name'] = chosen_scene
            data_point['init_object'] = init_object_location
            data_point['goal_object'] = goal_object_location
            # data_point["initial_agent_pose"] = initial_agent_pose

            #TODO this needs to be fixed for test
            reachable_positions = self.stretch_reachable_positions[data_point['scene_name']]
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

# class RoboTHORStretchDiverseBringObjectTaskSampler(StretchDiverseBringObjectTaskSampler):
#     def next_task(
#             self, force_advance_scene: bool = False
#     ) :
#
#         TODO this is too entangled, double check that we have redefined all the functions we use
#
#         if self.env is None:
#             self.env = self._create_environment()
#
#         if self.max_tasks is not None and self.max_tasks <= 0:
#             return None
#
#         if self.sampler_mode != "train" and self.length <= 0:
#             return None
#
#         data_point = self.get_source_target_indices()
#
#         # scene_name = data_point["scene_name"]
#         init_object = data_point['init_object']
#         goal_object = data_point['goal_object']
#         # agent_state = data_point["initial_agent_pose"]
#
#         # assert init_object["scene_name"] == goal_object["scene_name"] == scene_name
#         # assert init_object['object_id'] != goal_object['object_id']
#
#         scene_name = data_point["scene_name"]
#         agent_state = data_point["initial_agent_pose"]
#         self.reset_scene(scene_name)
#
#         this_controller = self.env
#
#         event = this_controller.step(
#             dict(
#                 action="TeleportFull",
#                 standing=True,
#                 x=agent_state["position"]["x"],
#                 y=agent_state["position"]["y"],
#                 z=agent_state["position"]["z"],
#                 rotation=dict(
#                     x=agent_state["rotation"]["x"],
#                     y=agent_state["rotation"]["y"],
#                     z=agent_state["rotation"]["z"],
#                 ),
#                 horizon=agent_state["cameraHorizon"],
#             )
#         )
#
#         if not event.metadata['lastActionSuccess']:
#             print('ERROR: Teleport failed')
#
#
#         # should_visualize_goal_start = [
#         #     x for x in self.visualizers if issubclass(type(x), StretchBringObjImageVisualizer)
#         # ]
#
#         initial_object_info = self.env.get_object_by_id(init_object["object_id"])
#         initial_agent_location = self.env.controller.last_event.metadata["agent"]
#         initial_hand_state = self.env.get_absolute_hand_state()
#
#         # source_img_adr = get_random_query_image_file_name(scene_name,init_object['object_id'], self.query_image_dict)
#         # goal_img_adr = get_random_query_image_file_name(scene_name,goal_object['object_id'], self.query_image_dict)
#
#         # source_object_query, source_img_adr = get_random_query_image(scene_name,init_object['object_id'], self.query_image_dict)
#         # goal_object_query, goal_img_adr = get_random_query_image(scene_name,goal_object['object_id'], self.query_image_dict)
#         # source_object_query_feature = get_random_query_feature_from_img_adr(source_img_adr)
#         # goal_object_query_feature = get_random_query_feature_from_img_adr(goal_img_adr)
#
#
#         task_info = {
#             'source_object_id': init_object['object_id'],
#             'goal_object_id': goal_object['object_id'],
#             "init_location": init_object,
#             "goal_location": goal_object,
#             'agent_initial_state': initial_agent_location,
#             'initial_object_location':initial_object_info,
#             'initial_hand_state': initial_hand_state,
#             # 'source_object_query': source_object_query,
#             # 'goal_object_query': goal_object_query,
#             # 'source_object_query_feature': source_object_query_feature,
#             # 'goal_object_query_feature': goal_object_query_feature,
#             # 'source_object_query_file_name' : source_img_adr,
#             # 'goal_object_query_file_name' : goal_img_adr,
#             'episode_number': random.uniform(0, 10000),
#         }
#
#
#         # if len(should_visualize_goal_start) > 0:
#         #     task_info["visualization_source"] = init_object
#         #     task_info["visualization_target"] = goal_object
#
#         self._last_sampled_task = self.TASK_TYPE(
#             env=self.env,
#             sensors=self.sensors,
#             task_info=task_info,
#             max_steps=self.max_steps,
#             action_space=self._action_space,
#             visualizers=self.visualizers,
#             reward_configs=self.rewards_config,
#         )
#
#         return self._last_sampled_task
#
#     def get_source_target_indices(self):
#
#         TODO this needs to be fixed for test
#
#         # super(type(self), self).get_source_target_indices()
#         data_point = {}
#         TODO this can't be more wrong
#         POSSIBLE_SCENES = ROBOTHOR_TRAIN
#         data_point['scene_name'] = random.choice(POSSIBLE_SCENES)
#         self.env.reset(data_point['scene_name'])
#         reachable_positions = self.env.reachable_points_with_rotations_and_horizons()
#         # reachable_positions = self.stretch_reachable_positions[data_point['scene_name']]
#         agent_pose = random.choice(reachable_positions)
#         source_obj, goal_obj = random.sample(self.objects, 2)
#         source_obj = self.env.closest_object_of_type(source_obj)
#         goal_obj = self.env.closest_object_of_type(goal_obj)
#
#         def convert(obj_dict):
#             obj_dict['object_id'] = obj_dict['objectId']
#             obj_dict['object_location'] = obj_dict['position']
#             return obj_dict
#
#         source_obj = convert(source_obj)
#         goal_obj = convert(goal_obj)
#
#         data_point['init_object'] = source_obj
#         data_point['goal_object'] = goal_obj
#
#         data_point['initial_agent_pose'] = {
#             "name": "agent",
#             "position": dict(x=agent_pose['x'], y=agent_pose['y'], z=agent_pose['z']),
#             "rotation": dict(x=0, y=agent_pose['rotation'], z=0),
#             "cameraHorizon": 20,
#             "isStanding": True,
#         }
#
#         return data_point