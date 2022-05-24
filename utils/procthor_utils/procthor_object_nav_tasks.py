from ast import For
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal

import gym
import numpy as np
import torch
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.cache_utils import DynamicDistanceCache
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from allenact.utils.system import get_logger

from utils.procthor_utils.procthor_helper import distance_to_object_id, position_dist, spl_metric
from utils.stretch_utils.stretch_constants import (
    MOVE_AHEAD,
    ROTATE_RIGHT,
    ROTATE_LEFT,
    DONE, 
    MOVE_BACK, 
    ROTATE_RIGHT_SMALL, 
    ROTATE_LEFT_SMALL
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
# from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment

from manipulathor_utils.debugger_util import ForkedPdb

class ObjectNavTask(Task[ManipulaTHOREnvironment]):
    _actions = (
        MOVE_AHEAD,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        DONE,
    )

    def __init__(
        self,
        env: ManipulaTHOREnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_config: Dict[str, Any],
        distance_cache: DynamicDistanceCache,
        distance_type: str = "geo",
        additional_visualize: Optional[bool] = None,
        visualizers: List[LoggerVisualizer] = [],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps,
            **kwargs,
        )
        self.env = env
        self.reward_config = reward_config
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.mirror = task_info["mirrored"]

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [
            self.env.last_event.metadata["agent"]["position"]
        ]
        self.task_info["taken_actions"] = []
        self.task_info["action_successes"] = []
        self.agent_body_dist_to_obj = []


        self.distance_cache = distance_cache

        self.distance_type = distance_type
        if distance_type == "geo":
            self.dist_to_target_func = self.min_geo_distance_to_target
        elif distance_type == "l2":
            self.dist_to_target_func = self.min_l2_distance_to_target
        else:
            raise NotImplementedError

        self.last_distance = self.dist_to_target_func()
        self.optimal_distance = self.last_distance
        self.closest_distance = self.last_distance

        self.visualizers = visualizers
        self.start_visualize()

        self.additional_visualize = (
            additional_visualize
            if additional_visualize is not None
            else (self.task_info["mode"] == "eval" or random.random() < 1 / 1000)
        )
        self.observations = [self.env.last_event.frame]
        self._metrics = None
        ForkedPdb().set_trace()

    def min_l2_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.
        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = float("inf")
        obj_id_to_obj_pos = {o["objectId"]: o["axisAlignedBoundingBox"]["center"] 
                                for o in self.env.last_event.metadata["objects"]}
        for object_id in self.task_info["target_object_ids"]:
            min_dist = min(
                min_dist,
                self.env.position_dist(
                    obj_id_to_obj_pos[object_id],
                    self.env.last_event.metadata["agent"]["position"],
                ),
            )
        if min_dist == float("inf"):
            get_logger().error(
                f"No target object {self.task_info['object_type']} found"
                f" in house {self.task_info['house_name']}."
            )
            return -1.0
        return min_dist

    def min_geo_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.
        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = None
        for object_id in self.task_info["target_object_ids"]:
            geo_dist = distance_to_object_id(
                env=self.env,
                distance_cache=self.distance_cache,
                object_id=object_id,
                house_name=self.task_info["house_name"],
            )
            if (min_dist is None and geo_dist >= 0) or (
                geo_dist >= 0 and geo_dist < min_dist
            ):
                min_dist = geo_dist
        if min_dist is None:
            return -1.0
        return min_dist
    
    
    def start_visualize(self):
        for visualizer in self.visualizers:
            if not visualizer.is_empty():
                print("OH NO VISUALIZER WAS NOT EMPTY")
                visualizer.finish_episode(self.env, self, self.task_info)
                visualizer.finish_episode_metrics(self, self.task_info, None)
            if type(visualizer)==StretchObjNavImageVisualizer:
                visualizer.log(self.env, "", obs=self.get_observations())
            else:
                visualizer.log(self.env, "")


    def visualize(self, action_str):
        for visualizer in self.visualizers:
            if type(visualizer)==StretchObjNavImageVisualizer:
                visualizer.log(self.env, action_str, obs=self.get_observations())
            else:
                visualizer.log(self.env, action_str)

    def finish_visualizer(self, episode_success):

        for visualizer in self.visualizers:
            visualizer.finish_episode(self.env, self, self.task_info)

    def finish_visualizer_metrics(self, metric_results):

        for visualizer in self.visualizers:
            visualizer.finish_episode_metrics(self, self.task_info, metric_results)

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()
    
    def render(
        self, mode: Literal["rgb", "depth"] = "rgb", *args, **kwargs
    ) -> np.ndarray:
        if mode == "rgb":
            frame = self.env.last_event.frame.copy()
        elif mode == "depth":
            frame = self.env.last_event.depth_frame.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if self.mirror:
            frame = np.fliplr(frame)

        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            obj
            for obj in self.env.last_event.metadata["objects"]
            if obj["visible"] and obj["objectType"] == self.task_info["object_type"]
        )

    def shaping(self) -> float:
        if self.reward_config['shaping_weight'] == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

        if self.distance_type == "l2":
            reward = max(self.closest_distance - cur_distance, 0)
            self.closest_distance = min(self.closest_distance, cur_distance)

            return reward * self.reward_config['shaping_weight']
        else:
            # Ensuring the reward magnitude is not greater than the total distance moved
            max_reward_mag = 0.0
            if len(self.path) >= 2:
                p0, p1 = self.path[-2:]
                max_reward_mag = position_dist(p0, p1, ignore_y=True)

            if (
                self.reward_config['positive_only_reward']
                and cur_distance is not None
                and cur_distance > 0.5
            ):
                reward = max(self.closest_distance - cur_distance, 0)
            elif self.last_distance is not None and cur_distance is not None:
                reward += self.last_distance - cur_distance

            self.last_distance = cur_distance
            self.closest_distance = min(self.closest_distance, cur_distance)

            return (
                max(
                    min(reward, max_reward_mag),
                    -max_reward_mag,
                )
                * self.reward_config['shaping_weight']
            )
    
    def get_observations(self, **kwargs) -> Any:
        obs = super().get_observations()
        if self.mirror:
            for o in obs:
                if ("rgb" in o or "depth" in o) and isinstance(obs[o], np.ndarray):
                    obs[o] = np.fliplr(obs[o])
        return obs

    def task_callback_data(self) -> Optional[Dict[str, Any]]:
        if not self.additional_visualize:
            return None

        # NOTE: Create top-down trajectory path visualization
        agent_path = [
            dict(x=p["x"], y=0.25, z=p["z"])
            for p in self._metrics["task_info"]["followed_path"]
        ]
        if not self.env.last_event.third_party_camera_frames:
            # assumes this is the only third party camera
            event = self.env.step({"action": "GetMapViewCameraProperties"})
            cam = event.metadata["actionReturn"].copy()
            cam["orthographicSize"] += 1
            self.env.step(
                {"action": "AddThirdPartyCamera", "skyboxColor":"white", **cam}
            )
        event = self.env.step({"action": "VisualizePath", "positions":agent_path})
        self.env.step({"action":"HideVisualizedPath"})

        return {
            "observations": self.observations,
            "path": event.third_party_camera_frames[0],
            **self._metrics,
        }

    def metrics(self) -> Dict[str, Any]:
        if self.is_done():
            result={} # placeholder for future
            metrics = super().metrics()
            metrics["dist_to_target"] = self.dist_to_target_func()
            metrics["total_reward"] = np.sum(self._rewards)
            metrics["spl"] = spl_metric(
                success=self._success,
                optimal_distance=self.optimal_distance,
                travelled_distance=self.travelled_distance,
            )
            metrics["success"] = self._success
            result["success"] = self._success
            self.finish_visualizer_metrics(result)
            self.finish_visualizer(self._success)

            self._metrics = metrics
            return metrics
        else:
            return {}
    
    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]
        ForkedPdb().set_trace()

        if self.mirror:
            if action_str == "RotateRight":
                action_str = "RotateLeft"
            elif action_str == "RotateLeft":
                action_str = "RotateRight"

        self.task_info["taken_actions"].append(action_str)

        if action_str == "Done":
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
            self.task_info["action_successes"].append(True)
        else:
            action_dict = {"action": action_str}
            self.env.step(action_dict)
            self.last_action_success = bool(self.env.last_event)

            position = self.env.last_event.metadata["agent"]["position"]
            self.path.append(position)
            self.task_info["followed_path"].append(position)
            self.task_info["action_successes"].append(self.last_action_success)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )
        
        obj_id_to_obj_pos = {o["objectId"]: o["axisAlignedBoundingBox"]["center"] 
                                for o in self.env.last_event.metadata["objects"]}
        self.agent_body_dist_to_obj.append(self.env.position_dist(
                    obj_id_to_obj_pos[self.task_info["target_object_ids"][0]],
                    self.env.last_event.metadata["agent"]["position"],
                ))

        if self.additional_visualize:
            self.observations.append(self.env.last_event.frame)
        self.visualize(action_str)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    
    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_config['step_penalty']

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config['goal_success_reward']
            else:
                reward += self.reward_config['failed_stop_reward']
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config['reached_horizon_reward']

        self._rewards.append(float(reward))
        return float(reward)



class StretchObjectNavTask(ObjectNavTask):
    _actions = (
        MOVE_AHEAD,
        MOVE_BACK,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        # ROTATE_RIGHT_SMALL,
        # ROTATE_LEFT_SMALL,
        DONE,
    )

class ExploreWiseObjectNavTask(ObjectNavTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_locations = [[k['x'], k['y'], k['z']] for k in (self.env.get_reachable_positions())]
        self.all_reachable_positions = torch.Tensor(all_locations)
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
        assert len(self.all_reachable_positions) > 0, 'no reachable positions to calculate reward'

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = self.reward_configs["step_penalty"]

        current_agent_location = self.env.get_agent_location()
        current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
        all_distances = self.all_reachable_positions - current_agent_location
        all_distances = (all_distances ** 2).sum(dim=-1)
        location_index = torch.argmin(all_distances)
        if self.has_visited[location_index] == 0:
            visited_new_place = True
        else:
            visited_new_place = False
        self.has_visited[location_index] = 1

        if visited_new_place:
            reward += self.reward_configs["exploration_reward"]
        
        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config['goal_success_reward']
            else:
                reward += self.reward_config['failed_stop_reward']
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config['reached_horizon_reward']
        self._rewards.append(float(reward))
        return float(reward)
        





