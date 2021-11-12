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
from ithor_arm.ithor_arm_constants import transport_wrapper
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_tasks import (
    AbstractPickUpDropOffTask,
)
from ithor_arm.ithor_arm_viz import LoggerVisualizer, BringObjImageVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.visualization_stuff_for_qualitative import TASKINFO
from utils.manipulathor_data_loader_utils import get_random_query_image, get_random_query_feature, get_random_query_feature_from_img_adr, get_random_query_image_file_name


class BringObjectAbstractTaskSampler(TaskSampler):

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
        env = ManipulaTHOREnvironment(
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


class DiverseBringObjectTaskSampler(BringObjectAbstractTaskSampler):
    # _TASK_TYPE = BringObjectTask
    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        possible_initial_locations = (
            "datasets/apnd-dataset/valid_agent_initial_locations.json"
        )

        with open(possible_initial_locations) as f:
            self.possible_agent_reachable_poses = json.load(f)

        self.query_image_dict = self.find_all_query_objects()
        self.all_possible_points = {}
        for scene in self.scenes:
            for object in self.objects:
                valid_position_adr = "datasets/apnd-dataset/pruned_object_positions/pruned_v3_valid_{}_positions_in_{}.json".format(
                    object, scene
                )
                try:
                    with open(valid_position_adr) as f:
                        data_points = json.load(f)
                except Exception:
                    print("Failed to load", valid_position_adr)
                    ForkedPdb().set_trace()
                    continue

                # if this is too big I can live with not having it and randomly sample each time
                all_locations_matrix = torch.tensor([[d['object_location']['x'], d['object_location']['y'], d['object_location']['z']] for d in data_points[scene]])
                self.all_possible_points[(scene, object)] = dict(
                    data_point_dict=data_points[scene],
                    data_point_matrix=all_locations_matrix
                )


        scene_names = set([x[0] for x in self.all_possible_points.keys()])

        if len(set(scene_names)) < len(self.scenes):
            print("Not all scenes appear")

        len_all_data_points = [len(v['data_point_dict']) for v in self.all_possible_points.values()]
        print(
            "Len dataset",
            sum(len_all_data_points),
        )

        if self.sampler_mode == "test":
            self.sampler_index = 0
            self.all_test_tasks = []

            #TODO we need to fix this later
            small_objects = ['Spatula', 'Egg']

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
            scene_name=scene_name, agentMode="arm", agentControllerType="mid-level"
        )

    def find_all_query_objects(self):
        IMAGE_DIR = 'datasets/apnd-dataset/query_images/'
        all_object_types = [f.split('/')[-1] for f in glob.glob(os.path.join(IMAGE_DIR, '*'))]
        all_possible_images = {object_type: [f for f in glob.glob(os.path.join(IMAGE_DIR, object_type, '*.png'))] for object_type in all_object_types}
        return all_possible_images


    def next_task(
            self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:

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

        assert init_object["scene_name"] == goal_object["scene_name"] == scene_name
        assert init_object['object_id'] != goal_object['object_id']

        self.reset_scene(scene_name)



        event1, event2, event3 = initialize_arm(self.env.controller)

        this_controller = self.env

        def put_object_in_location(location_point):

            object_id = location_point['object_id']
            location = location_point['object_location']
            event = transport_wrapper(
                this_controller,
                object_id,
                location,
            )
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
                horizon=agent_state["cameraHorizon"],
            )
        )

        if not event.metadata['lastActionSuccess']:
            print('ERROR: Teleport failed')

            # print(this_controller.last_event.metadata['sceneName'])
            # print(dict(
            #     action="TeleportFull",
            #     standing=True,
            #     x=agent_state["position"]["x"],
            #     y=agent_state["position"]["y"],
            #     z=agent_state["position"]["z"],
            #     rotation=dict(
            #         x=agent_state["rotation"]["x"],
            #         y=agent_state["rotation"]["y"],
            #         z=agent_state["rotation"]["z"],
            #     ),
            #     horizon=agent_state["cameraHorizon"],
            # ))
            # print(this_controller.last_event)


        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), BringObjImageVisualizer)
        ]

        initial_object_info = self.env.get_object_by_id(init_object["object_id"])
        initial_agent_location = self.env.controller.last_event.metadata["agent"]
        initial_hand_state = self.env.get_absolute_hand_state()

        # source_img_adr = get_random_query_image_file_name(scene_name,init_object['object_id'], self.query_image_dict)
        # goal_img_adr = get_random_query_image_file_name(scene_name,goal_object['object_id'], self.query_image_dict)

        source_object_query, source_img_adr = get_random_query_image(scene_name,init_object['object_id'], self.query_image_dict)
        goal_object_query, goal_img_adr = get_random_query_image(scene_name,goal_object['object_id'], self.query_image_dict)
        source_object_query_feature = get_random_query_feature_from_img_adr(source_img_adr)
        goal_object_query_feature = get_random_query_feature_from_img_adr(goal_img_adr)


        task_info = {
            'source_object_id': init_object['object_id'],
            'goal_object_id': goal_object['object_id'],
            "init_location": init_object,
            "goal_location": goal_object,
            'agent_initial_state': initial_agent_location,
            'initial_object_location':initial_object_info,
            'initial_hand_state': initial_hand_state,
            'source_object_query': source_object_query,
            'goal_object_query': goal_object_query,
            'source_object_query_feature': source_object_query_feature,
            'goal_object_query_feature': goal_object_query_feature,
            # 'source_object_query_file_name' : source_img_adr,
            # 'goal_object_query_file_name' : goal_img_adr,
            'episode_number': random.uniform(0, 10000),
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
            initial_location = torch.tensor([init_object_location['object_location']['x'], init_object_location['object_location']['y'], init_object_location['object_location']['z']])

            #calulcate distance of initial object location to all the possible target location
            all_goal_object_locations = self.all_possible_points[(chosen_scene, goal_obj)]['data_point_matrix']
            all_distances = (all_goal_object_locations - initial_location).norm(2, dim=-1)

            #randomly choosing a target location which is far enough from the initial location therefore chances of collisions and failures are low
            valid_goal_indices = torch.nonzero(all_distances > 1.0)
            chosen_goal_instance = random.choice(valid_goal_indices)
            goal_object_location = self.all_possible_points[(chosen_scene, goal_obj)]['data_point_dict'][chosen_goal_instance]

            selected_agent_init_loc = random.choice(
                self.possible_agent_reachable_poses[chosen_scene]
            )
            initial_agent_pose = {
                "name": "agent",
                "position": {
                    "x": selected_agent_init_loc["x"],
                    "y": selected_agent_init_loc["y"],
                    "z": selected_agent_init_loc["z"],
                },
                "rotation": {
                    "x": -0.0,
                    "y": selected_agent_init_loc["rotation"],
                    "z": 0.0,
                },
                "cameraHorizon": selected_agent_init_loc["horizon"],
                "isStanding": True,
            }

            data_point = {}
            data_point['scene_name'] = chosen_scene
            data_point['init_object'] = init_object_location
            data_point['goal_object'] = goal_object_location
            data_point["initial_agent_pose"] = initial_agent_pose

        else:

            task = self.all_test_tasks[self.sampler_index]
            self.sampler_index += 1

            #TODO_KIANA_ADDED
            task_details = TASKINFO['task_info_metrics']['task_info']
            task['scene_name'] = task_details['init_location']['scene_name']
            task['init_object'] = task_details['init_location']
            task['goal_object'] = task_details['goal_location']
            task['initial_agent_pose'] = task_details['agent_initial_state']



            data_point = task

        return data_point


class DiverseBringObjectTaskSamplerWRandomization(DiverseBringObjectTaskSampler):
    def reset_scene(self, scene_name):
        self.env.reset( scene_name=scene_name, agentMode="arm", agentControllerType="mid-level" )
        self.env.step(dict(action="RandomizeMaterials"))
        self.env.step(dict(action="RandomizeLighting"))


# class WDoneDiverseBringObjectTaskSampler(DiverseBringObjectTaskSampler):
#     # _TASK_TYPE = WPickUpBringObjectTask
#
# class WPickupAndExploreBOTS(DiverseBringObjectTaskSampler):
#     # _TASK_TYPE = WPickUPExploreBringObjectTask
#
# class NoPickupExploreBOTS(DiverseBringObjectTaskSampler):
#     _TASK_TYPE = NoPickUPExploreBringObjectTask