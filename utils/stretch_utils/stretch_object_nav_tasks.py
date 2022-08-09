import os
import json
import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal

import signal

import gym
import numpy as np
import pandas as pd
import torch
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.cache_utils import DynamicDistanceCache
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from utils.stretch_utils.stretch_visualizer import StretchObjNavImageVisualizer
from allenact.utils.system import get_logger
from moviepy.editor import ImageSequenceClip
from PIL import Image
import matplotlib.pyplot as plt

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

from manipulathor_utils.debugger_util import ForkedPdb
from datetime import datetime

def handler(signum, frame):
    raise Exception("Controller call took too long")

def unnormalize_image(img):
    # img = img.squeeze(0).squeeze(0)
    mean=np.array([0.48145466, 0.4578275, 0.40821073])
    std=np.array([0.26862954, 0.26130258, 0.27577711])
    img = (img * std + mean)
    img = np.clip(img, 0, 1) * 255
    return img.astype('uint8')

def get_true_sensor_obs(obs):
    if 'rgb_lowres' in obs:
        viz_image = obs['rgb_lowres']
        viz_image = unnormalize_image(viz_image)
        list_of_visualizations = [viz_image]
    if 'rgb_lowres_arm' in obs:
        kinect_image = obs['rgb_lowres_arm']
        kinect_image = unnormalize_image(kinect_image)
        list_of_visualizations.append(kinect_image)
    combined = np.concatenate(list_of_visualizations, axis=1)
    return combined


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

        pose = copy.deepcopy(self.env.last_event.metadata["agent"]["position"])
        pose["rotation"] = self.env.last_event.metadata["agent"]["rotation"]["y"]
        pose["horizon"] = self.env.last_event.metadata["agent"]["cameraHorizon"]
        self.task_info["followed_path"] = [pose]
        self.task_info["taken_actions"] = []
        self.task_info["action_successes"] = []
        self.task_info["rewards"] = []
        self.task_info["dist_to_target"] = []
        self.agent_body_dist_to_obj = []


        self.distance_cache = distance_cache

        self.distance_type = distance_type
        if distance_type == "geo":
            self.dist_to_target_func = self.min_geo_distance_to_target
        elif distance_type == "l2":
            self.dist_to_target_func = self.min_l2_distance_to_target
        elif distance_type == "real_world":
            self.dist_to_target_func = self.dummy_distance_to_target # maybe placeholder here for estimation later
        else:
            raise NotImplementedError

        self.last_distance = self.dist_to_target_func()
        self.optimal_distance = self.last_distance
        self.closest_distance = self.last_distance
        self.task_info["dist_to_target"].append(self.last_distance)

        self.visualizers = visualizers
        self.start_visualize()

        self.additional_visualize = (
            additional_visualize
            if additional_visualize is not None
            else (self.task_info["mode"] == "eval" )#or random.random() < 1 / 1000)
        )
        self.observations = [get_true_sensor_obs(self.get_observations())]#[self.env.last_event.frame]
        self._metrics = None

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
    
    def dummy_distance_to_target(self) -> float:
        return float("inf")
    
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
            and self.dist_to_target_func() < self.task_info['success_distance'] 
        )
    
    def calc_action_stat_metrics(self) -> Dict[str, Any]:
        action_stat = {
            "metric/action_stat/" + action_str: 0.0 for action_str in self._actions
        }
        action_success_stat = {
            "metric/action_success/" + action_str: 0.0 for action_str in self._actions
        }
        action_success_stat["metric/action_success/total"] = 0.0

        seq_len = len(self.task_info["taken_actions"])

        for (action_name, action_success) in list(zip(
                            self.task_info["taken_actions"],
                            self.task_info["action_successes"])):
            action_stat["metric/action_stat/" + action_name] += 1.0

            action_success_stat[
                "metric/action_success/{}".format(action_name)
            ] += action_success
            action_success_stat["metric/action_success/total"] += action_success

        action_success_stat["metric/action_success/total"] /= seq_len

        for action_name in self._actions:
            action_success_stat[
                "metric/" + "action_success/{}".format(action_name)
            ] /= (action_stat["metric/action_stat/" + action_name] + 0.000001)
            action_stat["metric/action_stat/" + action_name] /= seq_len

        result = {**action_stat, **action_success_stat}
        return result

    def shaping(self) -> float:
        cur_distance = self.dist_to_target_func()
        self.task_info["dist_to_target"].append(cur_distance)

        if self.reward_config['shaping_weight'] == 0.0:
            return 0

        reward = 0.0

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


    def metrics(self) -> Dict[str, Any]:
        if self.is_done():
            result={} # placeholder for future
            metrics = super().metrics()
            if self.distance_type != "real_world":
                metrics = {**metrics, **self.calc_action_stat_metrics()}
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
            sr = self.env.step(action_dict)
            # RH NOTE: this is an important change from procthor because of how the noise model 
            # is implemented. the env step return corresponds to the nominal/called 
            # action, which may or may not be the last thing the controller did.
            self.last_action_success = bool(sr.metadata["lastActionSuccess"])

            position = self.env.last_event.metadata["agent"]["position"]
            self.path.append(position)
            pose = copy.deepcopy(
                self.env.last_event.metadata["agent"]["position"]
            )
            pose["rotation"] = self.env.last_event.metadata["agent"]["rotation"][
                "y"
            ]
            pose["horizon"] = self.env.last_event.metadata["agent"][
                "cameraHorizon"
            ]
            self.task_info["followed_path"].append(pose)            
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
            self.observations.append(get_true_sensor_obs(self.get_observations()))

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
        self.task_info["rewards"].append(float(reward))
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

class StretchNeckedObjectNavTask(ObjectNavTask):
    _actions = (
        MOVE_AHEAD,
        # MOVE_BACK,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        "LookUp",
        "LookDown",
        DONE,
    )

class StretchNeckedObjectNavTaskUpdateOrder(ObjectNavTask):
    # for evaluating weights from Matt
    _actions = (
        MOVE_AHEAD,
        ROTATE_LEFT,
        ROTATE_RIGHT,
        DONE,
        "LookUp",
        "LookDown"
    )

class StretchObjectNavTaskKinectSegmentationSuccess(StretchObjectNavTask):
    def _is_goal_in_range(self) -> bool:
        all_kinect_masks = self.env.controller.last_event.third_party_instance_masks[0]
        for object_id in self.task_info["target_object_ids"]:
            if object_id in all_kinect_masks and self.dist_to_target_func() < self.task_info['success_distance']:
                return True
        
        return False
    
class StretchObjectNavTaskIntelSegmentationSuccess(StretchObjectNavTask):
    def _is_goal_in_range(self) -> bool:
        all_intel_masks = self.env.controller.last_event.instance_masks
        for object_id in self.task_info["target_object_ids"]:
            if object_id in all_intel_masks and self.dist_to_target_func() < self.task_info['success_distance']:
                return True
        
        return False

class StretchObjectNavTaskSegmentationSuccessActionFail(StretchObjectNavTaskKinectSegmentationSuccess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recent_three_strikes = 0
    
    def _step(self, action: int) -> RLStepResult:
        sr = super()._step(action)

        if self.last_action_success or self._took_end_action:
            self.recent_three_strikes = 0
            return sr
        elif self.recent_three_strikes < 2:
            self.recent_three_strikes += 1
            return sr
        else:
            # print('Task ended for repeated action failure')
            # ForkedPdb().set_trace()
            self._took_end_action = True
            step_result = RLStepResult(
                observation=sr.observation,
                reward=sr.reward - self.reward_config['got_stuck_penalty'],
                done=self.is_done(),
                info={"last_action_success": self.last_action_success, "action": action},
            )
            return step_result
    
    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_config['step_penalty']
        if not self.last_action_success:
            reward += self.reward_config['failed_action_penalty']

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config['goal_success_reward']
            else:
                reward += self.reward_config['failed_stop_reward']
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config['reached_horizon_reward']

        self._rewards.append(float(reward))
        self.task_info["rewards"].append(float(reward))
        return float(reward)



class ExploreWiseObjectNavTask(StretchObjectNavTaskSegmentationSuccessActionFail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_locations = [[k['x'], k['y'], k['z']] for k in (self.env.get_reachable_positions())]
        self.all_reachable_positions = torch.Tensor(all_locations)
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
        assert len(self.all_reachable_positions) > 0, 'no reachable positions to calculate reward'

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = self.reward_config["step_penalty"]

        # additional scaling step penalty, thresholds at half max steps at the failed action penalty
        reward += -0.2 * (1.1)**(np.min([-(self.max_steps/2)+self.num_steps_taken(),0])) 

        if not self.last_action_success:
            reward += self.reward_config['failed_action_penalty']

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
            reward += self.reward_config["exploration_reward"]
                
        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config['goal_success_reward']
            else:
                reward += 2*self.reward_config['failed_action_penalty']
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config['reached_horizon_reward']
        self._rewards.append(float(reward))
        self.task_info["rewards"].append(float(reward))
        return float(reward)
    
    def metrics(self) -> Dict[str, Any]:
        result = super(ExploreWiseObjectNavTask, self).metrics()

        if self.is_done():
            result['percent_room_visited'] = self.has_visited.mean().item()
            result['new_locations_visited'] = self.has_visited.sum().item()
        return result
        

class RealStretchObjectNavTask(StretchObjectNavTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time = datetime.now()
        self.last_time = None
        signal.signal(signal.SIGALRM, handler)
        self.additional_visualize = True
        self.manual_action = False # can set later in a breakpoint if needed
    
    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]
        print('Model Said', action_str, ' as action ', str(self.num_steps_taken()))

        # self.manual_action = True
        # self.env.kinect_depth
        if self.manual_action:
            ACTIONS_ORDERED = [MOVE_AHEAD,MOVE_BACK,ROTATE_RIGHT,ROTATE_LEFT,DONE]
            SHORTENED_ACTIONS_ORDERED = ['m','b','r','l','d']
            action = ''
            while(True):
                print('Agent state')
                print(self.env.controller.last_event.metadata)
                ForkedPdb().set_trace()

                if action == '':
                    action_str = action_str #Just use model's prediction
                    break
                try:
                    action_str = ACTIONS_ORDERED[SHORTENED_ACTIONS_ORDERED.index(action)]
                    break
                except Exception:
                    print("wrong action")
                    continue

        print('Action Called', action_str)

        # add/remove/adjust to allow graceful exit from auto-battlebots
        end_ep_early = False
        if self.num_steps_taken() % 10 == 0:
            print('verify continue? set end_ep_early=True to gracefully fail out or activate self.manual_action')
            ForkedPdb().set_trace()

        self._last_action_str = action_str
        action_dict = {"action": action_str}
        signal.alarm(8) # second timeout - catch missed server connection. THIS IS NOT THREAD SAFE
        try:
            self.env.step(action_dict)
        except:
            print('Controller call took too long, continue to try to continue or set end_ep_early to fail out')
            ForkedPdb().set_trace()
            self.env.step({"action": "Done"})
        
        signal.alarm(0)

        obj_dis = self.dist_to_target_func()

        if action_str == "Done" or end_ep_early:
            self._took_end_action = True
            dt_total = (datetime.now() - self.start_time).total_seconds()/60
            print('I think I found a ', self.task_info['object_type'], ' after ', str(dt_total), ' minutes.' )
            print('Was I correct? Set self._success in trace. Set obj_dis in meters from my spine. Default false/Inf.')            
            ForkedPdb().set_trace()
        
        nominal = self.env.nominal_agent_location
        position = dict(x=nominal["x"], y=nominal["y"], z=nominal["z"], )
        self.path.append(position)
        
        pose = copy.deepcopy(nominal)
        self.task_info["followed_path"].append(pose) 
        if self.manual_action and action != '':
            self.task_info["taken_actions"].append("manual override")
        else:
            self.task_info["taken_actions"].append(action_str)
        self.task_info["action_successes"].append("unknown")
        self.task_info["dist_to_target"].append(obj_dis)

        if self.last_time is not None:
            dt = (datetime.now() - self.last_time).total_seconds()
            print('FPS: ', str(1/dt))
        self.last_time = datetime.now()

        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        self.visualize(last_action_name)

        if True: #self.additional_visualize:
            self.observations.append(get_true_sensor_obs(self.get_observations()))

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result
    
    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = 0
        self.task_info["rewards"].append(float(reward))
        return reward
    
    def metrics(self) -> Dict[str, Any]:
        if self.is_done():
            # result={} # placeholder for future
            metrics = super().metrics()
            # metrics = {**metrics, **self.calc_action_stat_metrics()}
            metrics["dist_to_target"] = self.task_info["dist_to_target"][-1]
            metrics["total_reward"] = "Not computed for real"
            metrics["spl"] = "Not computed for real"
            metrics["success"] = self._success
            # result["success"] = self._success
            # self.finish_visualizer_metrics(result)
            # self.finish_visualizer(self._success)

            self._metrics = metrics
            return metrics
        else:
            return {}



