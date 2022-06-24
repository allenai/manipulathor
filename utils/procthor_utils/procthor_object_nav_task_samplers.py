from typing import Any, Dict, List, Optional
import platform
import pickle
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union


import gym
import datasets
import numpy as np
from ai2thor.controller import Controller
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger
from ithor_arm.ithor_arm_viz import LoggerVisualizer


from utils.procthor_utils.procthor_types import AgentPose, Vector3
from utils.stretch_utils.stretch_object_nav_tasks import StretchObjectNavTask
from utils.stretch_utils.stretch_constants import ADITIONAL_ARM_ARGS
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from scripts.stretch_jupyter_helper import get_relative_stretch_current_arm_state


from manipulathor_utils.debugger_util import ForkedPdb


class ProcTHORObjectNavTaskSampler(TaskSampler):

    def __init__(
        self,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        rewards_config: Dict,
        task_type: type,
        houses: datasets.Dataset,
        house_inds: List[int],
        target_object_types: List[str],
        resample_same_scene_freq: int,
        distance_type: str,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        visualizers: List[LoggerVisualizer] = [],
        *args,
        **kwargs
    ) -> None:
        self.TASK_TYPE = task_type
        self.rewards_config = rewards_config
        self.environment_type = env_args['environment_type']
        del env_args['environment_type']
        self.env_args = env_args
        
        self.grid_size = 0.25
        self.env: Optional[StretchManipulaTHOREnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = gym.spaces.Discrete(len(self.TASK_TYPE.class_action_names()))
        self.resample_same_scene_freq = resample_same_scene_freq

        self._last_sampled_task: Optional[Task] = None
        
        self.seed: Optional[int] = None
        if seed is not None:
            self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        
        self.visualizers = visualizers
        self.sampler_mode = kwargs["sampler_mode"]
        if self.sampler_mode != "train":
            self.rewards_config['shaping_weight'] = 0.0

        self.episode_index = 0
        self.houses = houses
        self.house_inds = house_inds
        self.house_inds_index = 0
            
        self.valid_rotations = [0,90,180,270]
        self.distance_type = distance_type
        self.distance_cache = DynamicDistanceCache(rounding=1)
        self.target_object_types_set = set(target_object_types)
        self.obj_type_counter = Counter(
            {obj_type: 0 for obj_type in target_object_types}
        )
        self.reachable_positions_map: Dict[int, Vector3] = dict()
        self.objects_in_scene_map: Dict[str, List[str]] = dict()
        self.visible_objects_cache = dict()
        self.max_tasks = max_tasks 
        self.reset_tasks = self.max_tasks
        
        self.max_vis_points = 6
        self.max_agent_positions = 6         
        self.p_greedy_target_object = 0.8
        self.min_raycast_distance = 1.5

        self.success_distance = 1.0

        self.reset()


    def set_seed(self, seed: int):
        set_seed(seed)
        
    def _create_environment(self, **kwargs) -> StretchManipulaTHOREnvironment:
        env = self.environment_type(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )
        return env

    def set_reachable_positions(self):
        pass

    @property
    def length(self) -> Union[int, float]:
        """Length.
        # Returns
        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[StretchObjectNavTask]:
        # NOTE: This book-keeping should be done in TaskSampler...
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

    def get_nearest_positions(self, world_position: Vector3) -> List[Vector3]:
        """Get the n reachable positions that are closest to the world_position."""
        self.reachable_positions.sort(
            key=lambda p: sum((p[k] - world_position[k]) ** 2 for k in ["x", "z"])
        )
        return self.reachable_positions[
            : min(
                len(self.reachable_positions),
                self.max_agent_positions,
            )
        ]

    def get_nearest_agent_height(self, y_coordinate: float) -> float:
        if self.env_args['agentMode'] == 'locobot':
            return 0.8697997
        elif self.env_args['agentMode'] == 'stretch':
            return 1.27 # to intel camera, measured physical
        else:
            return 1.5759992 # from default agent - is guess. TODO check stretch

    @property
    def house_index(self) -> int:
        return self.house_inds[self.house_inds_index]

    def is_object_visible(self, object_id: str) -> bool:
        """Return True if object_id is visible without any interaction in the scene.

        This method makes an approximation based on checking if the object
        is hit with a raycast from nearby reachable positions.
        """
        # NOTE: Check the cached visible objects first.
        if (
            self.house_index in self.visible_objects_cache
            and object_id in self.visible_objects_cache[self.house_index]
        ):
            return self.visible_objects_cache[self.house_index][object_id]
        elif self.house_index not in self.visible_objects_cache:
            self.visible_objects_cache[self.house_index] = dict()

        # NOTE: Get the visibility points on the object
        visibility_points = self.env.controller.step(
            action="GetVisibilityPoints", objectId=object_id, raise_for_failure=True
        ).metadata["actionReturn"]

        # NOTE: Randomly sample visibility points
        for vis_point in random.sample(
            population=visibility_points,
            k=min(len(visibility_points), self.max_vis_points),
        ):
            # NOTE: Get the nearest reachable agent positions to the target object.
            agent_positions = self.get_nearest_positions(world_position=vis_point)
            for agent_pos in agent_positions:
                agent_pos = agent_pos.copy()
                agent_pos["y"] = self.get_nearest_agent_height(
                    y_coordinate=vis_point["y"]
                )
                event = self.env.controller.step(
                    action="PerformRaycast",
                    origin=agent_pos,
                    destination=vis_point,
                )
                hit = event.metadata["actionReturn"]
                if (
                    event.metadata["lastActionSuccess"]
                    and hit["objectId"] == object_id
                    and hit["hitDistance"] < np.min([self.env_args['visibilityDistance'],self.min_raycast_distance])
                ):
                    self.visible_objects_cache[self.house_index][object_id] = True
                    return True

        self.visible_objects_cache[self.house_index][object_id] = False
        return False

    @property
    def target_objects_in_scene(self) -> Dict[str, List[str]]:
        """Return a map from the object type to the objectIds in the scene."""
        if self.house_index in self.objects_in_scene_map:
            return self.objects_in_scene_map[self.house_index]

        event = self.env.controller.step(action="ResetObjectFilter", raise_for_failure=True)
        objects = event.metadata["objects"]
        out = {}
        for obj in objects:
            if obj["objectType"] in self.target_object_types_set:
                if obj["objectType"] not in out:
                    out[obj["objectType"]] = []
                out[obj["objectType"]].append(obj["objectId"])
        self.objects_in_scene_map[self.house_index] = out
        return out

    def sample_target_object_ids(self) -> Tuple[str, List[str]]:
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
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)

                if visible_ids:
                    self.obj_type_counter[obj_type] += 1
                    return obj_type, visible_ids
        else:
            candidates = dict()
            for obj_type, object_ids in self.target_objects_in_scene.items():
                visible_ids = []
                for object_id in object_ids:
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)

                if visible_ids:
                    candidates[obj_type] = visible_ids

            if candidates:
                return random.choice(list(candidates.items()))

        raise ValueError(f"No target objects in house {self.house_index}.")

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.house_index]
    
    # def reset_scene(self):
    #     self.env.reset(
    #         scene_name='Procedural',
    #         agentMode=self.env_args['agentMode'], agentControllerType=self.env_args['agentControllerType']
    #     )    

    def increment_scene(self) -> bool:
        """Increment the current scene.
        Returns True if the scene works with reachable positions, False otherwise.
        """

        # self.reset_scene()
        self.increment_scene_index()

        # self.env.controller.step(action="DestroyHouse", raise_for_failure=True)
        # self.env.controller.reset()
        self.env.reset(scene_name='Procedural')
        
        self.env.list_of_actions_so_far = []
        self.house_entry = self.houses[self.house_index]
        self.house = pickle.loads(self.house_entry["house"])

        if platform.system() == "Darwin": #TODO remove
            print('The house is ', self.house_index)

        self.env.controller.step(
            action="CreateHouse", house=self.house, raise_for_failure=True
        )
        
        if self.house_index not in self.reachable_positions_map:
            pose = self.house["metadata"]["agent"].copy()
            if self.env_args['agentMode'] == 'locobot':
                del pose["standing"]
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
        
        # verify the stretch arm is stowed
        if self.env_args['agentMode'] == 'stretch':
            arm_pos = get_relative_stretch_current_arm_state(self.env.controller)
            assert abs(sum(arm_pos.values())) < 0.001

        return True
        

    def increment_scene_index(self):
        self.house_inds_index = (self.house_inds_index + 1) % len(self.house_inds)

    def next_task(self, force_advance_scene: bool = False) -> Optional[StretchObjectNavTask]:
        if self.env is None:
            self.env = self._create_environment()
        
        # NOTE: Stopping condition
        if self.max_tasks <= 0:
            return None

        # NOTE: determine if the house should be changed.
        if force_advance_scene or (
            self.resample_same_scene_freq > 0 
            and self.episode_index % self.resample_same_scene_freq == 0
        ):
            while not self.increment_scene():
                pass

        # NOTE: Choose target object
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                target_object_type, target_object_ids = self.sample_target_object_ids()
                break
            except ValueError:
                while not self.increment_scene():
                    pass

        if random.random() < self.env_args['p_randomize_material']:
            self.env.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.env.controller.step(action="ResetMaterials", raise_for_failure=True)

        self.env.controller.step(
            action="SetObjectFilter",
            objectIds=target_object_ids,
            raise_for_failure=True,
        )

        # NOTE: Set agent pose
        event = None
        attempts = 0
        while not event:
            attempts+=1
            starting_pose = AgentPose(
                position=random.choice(self.reachable_positions),
                rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
                horizon=0,
            )
            if self.env_args['agentMode'] != 'locobot':
                starting_pose['standing']=True
                starting_pose['horizon'] = self.env_args['horizon_init']
            event = self.env.controller.step(action="TeleportFull", **starting_pose)
            if attempts > 10:
                get_logger().error(f"Teleport failed {attempts-1} times in house {self.house_index} - something may be wrong")
            

        self.episode_index += 1
        self.max_tasks -= 1
        
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
                "mode": self.env_args['agentMode'],
                # "process_ind": self.process_ind,
                "scene_name": self.env_args['scene'],
                "house_name": str(self.house_index),
                "rooms": self.house_entry["rooms"],
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.env_args['allow_flipping'] and random.random() > 0.5,
                'success_distance': self.success_distance
            },
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.max_tasks = self.reset_tasks
        self.house_inds_index = 0



class RoboThorObjectNavTestTaskSampler(ProcTHORObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

    def next_task(self, force_advance_scene: bool = False) -> Optional[StretchObjectNavTask]:
        # ForkedPdb().set_trace()
        while True:
            # NOTE: Stopping condition
            if self.env is None:
                self.env = self._create_environment()
        
            # NOTE: Stopping condition
            if self.max_tasks <= 0:
                return None


            epidx = self.house_inds[self.max_tasks - 1]
            ep = self.houses[epidx]
            # ForkedPdb().set_trace()

            if self.last_scene is None or self.last_scene != ep["scene"]:
                self.last_scene = ep["scene"]
                # self.env.controller.reset(ep["scene"])
                self.env.reset(scene_name=ep["scene"])

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.env.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.env.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )
            if self.env_args['agentMode'] != 'locobot':
                ep["agentPose"]["standing"] = True
                ep["agentPose"]["horizon"] = self.env_args['horizon_init'] # reset for stretch agent
            event = self.env.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = self.TASK_TYPE(
                # visualize=self.episode_index in self.epids_to_visualize,
                env=self.env,
                sensors=self.sensors,
                max_steps=self.max_steps,
                reward_config=self.rewards_config,
                distance_type=self.distance_type,
                distance_cache=self.distance_cache,
                visualizers=self.visualizers,
                task_info={
                    "mode": self.env_args['agentMode'],
                    "scene_name": ep["scene"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__global{epidx}__{ep['targetObjectType']}",
                    'success_distance': self.success_distance,
                    **difficulty,
                },
            )

            self.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


