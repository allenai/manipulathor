import glob
import pickle
import random
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gym
import datasets
import numpy as np
from ai2thor.controller import Controller
# from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger
from ithor_arm.ithor_arm_viz import LoggerVisualizer

# from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from utils.procthor_utils.procthor_types import AgentPose, Vector3
from utils.procthor_utils.procthor_object_nav_tasks import ProcTHORObjectNavTask
from utils.stretch_utils.stretch_constants import ADITIONAL_ARM_ARGS
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment


from manipulathor_utils.debugger_util import ForkedPdb


class ProcTHORObjectNavTaskSampler(TaskSampler):

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

        # ROOMS_TO_USE = [int(scene.replace('ProcTHOR', '')) for scene in self.scenes]



        # self.dataset_files ={}
        # print('Load dataset')
        # dataset_files = 'datasets/procthor_apnd_dataset/room_id_'
        # # TODO we can open this on the fly
        # for room_ind in ROOMS_TO_USE:
        #     files = [f for f in glob.glob(dataset_files + str(room_ind) + '_*.json')] # TODO maybe it's better to do this only once
        #     if len(files) == 0:
        #         # print(room_ind, 'is missing')
        #         continue
        #     elif len(files) > 1:
        #         print(room_ind, 'multiple instance')
        #         f = random.choice(files)
        #     else:
        #         f = files[0]
        #     with open(f) as file_des:
        #         dict = json.load(file_des) # TODO maybe even convert everything into h5py?
        #         self.dataset_files[room_ind] = dict

        # print('Finished Loading data')

        # self.args_house_inds = list(self.dataset_files.keys())
        # random.shuffle(self.args_house_inds)
        # ForkedPdb().set_trace()


    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
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
    def last_sampled_task(self) -> Optional[ProcTHORObjectNavTask]:
        # NOTE: This book-keeping should be done in TaskSampler...
        return self._last_sampled_task

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()

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
                max_agent_positions=6,
            )
        ]

    def get_nearest_agent_height(self, y_coordinate: float) -> float:
        return 1.5759992 # from default agent

    @property
    def house_index(self) -> int:
        return self.args.house_inds[self.house_inds_index]

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
        visibility_points = self.controller.step(
            action="GetVisibilityPoints", objectId=object_id, raise_for_failure=True
        ).metadata["actionReturn"]

        # NOTE: Randomly sample visibility points
        for vis_point in random.sample(
            population=visibility_points,
            k=min(len(visibility_points), max_vis_points=6),
        ):
            # NOTE: Get the nearest reachable agent positions to the target object.
            agent_positions = self.get_nearest_positions(world_position=vis_point)
            for agent_pos in agent_positions:
                agent_pos = agent_pos.copy()
                agent_pos["y"] = self.get_nearest_agent_height(
                    y_coordinate=vis_point["y"]
                )
                event = self.controller.step(
                    action="PerformRaycast",
                    origin=agent_pos,
                    destination=vis_point,
                )
                hit = event.metadata["actionReturn"]
                if (
                    event.metadata["lastActionSuccess"]
                    and hit["objectId"] == object_id
                    and hit["hitDistance"] < self.env_args.visibility_distance
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

        event = self.controller.step(action="ResetObjectFilter", raise_for_failure=True)
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
        if random.random() < 0.8: # p_greedy_target_object
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

    def increment_scene(self) -> bool:
        """Increment the current scene.
        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        # self.controller.step(action="DestroyHouse", raise_for_failure=True)
        self.controller.reset()
        self.house_entry = self.args.houses[self.house_index]
        self.house = pickle.loads(self.house_entry["house"])

        self.controller.step(
            action="CreateHouse", house=self.house, raise_for_failure=True
        )

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            pose = self.house["metadata"]["agent"].copy()
            if self.args.controller_args["agentMode"] == "locobot":
                del pose["standing"]
            event = self.controller.step(action="TeleportFull", **pose)
            if not event:
                get_logger().warning(f"Initial teleport failing in {self.house_index}.")
                return False
            rp_event = self.controller.step(action="GetReachablePositions")
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
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args.house_inds)

    def next_task(self, force_advance_scene: bool = False) -> Optional[ProcTHORObjectNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = Controller(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.controller._build.commit_id}"
            )

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

        if False and random.random() < 0.8: #TODO
            self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.controller.step(action="ResetMaterials", raise_for_failure=True)

        self.controller.step(
            action="SetObjectFilter",
            objectIds=target_object_ids,
            raise_for_failure=True,
        )

        # NOTE: Set agent pose
        standing = (
            {}
            if self.args.controller_args["agentMode"] == "locobot"
            else {"standing": True}
        )
        starting_pose = AgentPose(
            position=random.choice(self.reachable_positions),
            rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
            horizon=30,
            **standing,
        )
        event = self.controller.step(action="TeleportFull", **starting_pose)
        if not event:
            get_logger().warning(
                f"Teleport failing in {self.house_index} at {starting_pose}"
            )

        self.episode_index += 1
        self.args.max_tasks -= 1

        self._last_sampled_task = ProcTHORObjectNavTask(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type=self.args.distance_type,
            distance_cache=self.distance_cache,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "house_name": str(self.house_index),
                "rooms": self.house_entry["rooms"],
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
            },
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.args.max_tasks = self.reset_tasks
        self.house_inds_index = 0
