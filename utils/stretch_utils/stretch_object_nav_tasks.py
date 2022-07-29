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
from manipulathor_baselines.stretch_object_nav_baselines.callbacks.local_logging import LocalLogging

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

        self.visualizers = visualizers
        self.start_visualize()

        self.additional_visualize = (
            additional_visualize
            if additional_visualize is not None
            else (self.task_info["mode"] == "eval" or random.random() < 1 / 1000)
        )
        self.observations = [self.env.last_event.frame]
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

    def task_callback_data(self) -> Optional[Dict[str, Any]]:
        if not self.additional_visualize:
            return None

        # NOTE: Create top-down trajectory path visualization
        agent_path = [
            dict(x=p["x"], y=0.25, z=p["z"])
            for p in self._metrics["task_info"]["followed_path"]
        ]
        # THIS ASSUMES BOTH CAMERAS ARE ON (slash only works for stretch with one third-party camera)
        if len(self.env.controller.last_event.third_party_camera_frames) < 2:
            event = self.env.step({"action": "GetMapViewCameraProperties"})
            cam = event.metadata["actionReturn"].copy()
            cam["orthographicSize"] += 1
            self.env.step(
                {"action": "AddThirdPartyCamera", "skyboxColor":"white", **cam}
            )
        event = self.env.step({"action": "VisualizePath", "positions":agent_path})
        self.env.step({"action":"HideVisualizedPath"})
        path = event.third_party_camera_frames[1]
        # ForkedPdb().set_trace()

        df = pd.read_csv(
            f"experiment_output/ac-data/{self.task_info['id']}.txt",
            names=list(self.class_action_names())+["EstimatedValue"],
            # names=[
            #     "MoveAhead",
            #     "RotateLeft",
            #     "RotateRight",
            #     "End",
            #     "LookUp",
            #     "LookDown",
            #     "EstimatedValue",
            # ],
        )
        ForkedPdb().set_trace()
        ep_length = self._metrics["ep_length"]

        # get returns from each step
        returns = []
        for r in reversed(self.task_info["rewards"]):
            if len(returns) == 0:
                returns.append(r)
            else:
                returns.append(r + returns[-1] * 0.99) # gamma value
        returns = returns[::-1]

        video_frames = []
        for step in range(self._metrics["ep_length"] + 1):
            is_first_frame = step == 0
            is_last_frame = step == self._metrics["ep_length"]

            agent_frame = np.array(
                Image.fromarray(self.observations[step]).resize((224, 224))
            )
            frame_number = step
            dist_to_target = self.task_info["dist_to_target"][step]

            if is_first_frame:
                last_action_success = None
                last_reward = None
                return_value = None
            else:
                last_action_success = self.task_info["action_successes"][step - 1]
                last_reward = self.task_info["rewards"][step - 1]
                return_value = returns[step - 1]

            if is_last_frame:
                action_dist = None
                critic_value = None

                taken_action = None
            else:
                policy_critic_value = df.iloc[step].values.tolist()
                action_dist = policy_critic_value[:6]
                critic_value = policy_critic_value[6]

                taken_action = self.task_info["taken_actions"][step]

            video_frame = LocalLogging.get_video_frame(
                agent_frame=agent_frame,
                frame_number=frame_number,
                last_reward=(
                    round(last_reward, 2) if last_reward is not None else None
                ),
                critic_value=(
                    round(critic_value, 2) if critic_value is not None else None
                ),
                return_value=(
                    round(return_value, 2) if return_value is not None else None
                ),
                dist_to_target=round(dist_to_target, 2),
                action_dist=action_dist,
                ep_length=ep_length,
                last_action_success=last_action_success,
                taken_action=taken_action,
            )
            video_frames.append(video_frame)

        for _ in range(9):
            video_frames.append(video_frames[-1])

        os.makedirs(f"trajectories/{self.task_info['id']}", exist_ok=True)

        imsn = ImageSequenceClip([frame for frame in video_frames], fps=10)
        imsn.write_videofile(f"trajectories/{self.task_info['id']}/frames.mp4")

        # save the top-down path
        Image.fromarray(path).save(f"trajectories/{self.task_info['id']}/path.png")

        # save the value function over time
        fig, ax = plt.subplots()
        estimated_values = df.EstimatedValue.to_numpy()
        ax.plot(estimated_values, label="Critic Estimated Value")
        ax.plot(returns, label="Return")
        ax.set_ylabel("Value")
        ax.set_xlabel("Time Step")
        ax.set_title("Value Function over Time")
        ax.legend()
        fig.savefig(
            f"trajectories/{self.task_info['id']}/value_fn.svg",
            bbox_inches="tight",
        )

        with open(f"trajectories/{self.task_info['id']}/data.json", "w") as f:
            json.dump(
                {
                    "id": self.task_info["id"],
                    "spl": self._metrics["spl"],
                    "success": self._metrics["success"],
                    "finalDistance": self.task_info["dist_to_target"][-1],
                    "initialDistance": self.task_info["dist_to_target"][0],
                    "minDistance": min(self.task_info["dist_to_target"]),
                    "episodeLength": self._metrics["ep_length"],
                    "confidence": (
                        None
                        if self.task_info["taken_actions"][-1] != "End"
                        else df.End.to_list()[-1]
                    ),
                    "failedActions": len(
                        [s for s in self.task_info["action_successes"] if not s]
                    ),
                    "targetObjectType": self.task_info["object_type"],
                    "numTargetObjects": len(self.task_info["target_object_ids"]),
                    "mirrored": self.task_info["mirrored"],
                    "scene": {
                        "name": self.task_info["house_name"],
                        "split": "train",
                        "rooms": 1,
                    },
                },
                f,
            )

        return {
            "observations": self.observations,
            "path": path,
            **self._metrics,
        }


    def metrics(self) -> Dict[str, Any]:
        if self.is_done():
            result={} # placeholder for future
            metrics = super().metrics()
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
            # TODO: does not include second camera.
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
    
    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]
        print('Model Said', action_str, ' as action ', str(self.num_steps_taken()))

        self.manual_action = False

        # add/remove/adjust to allow graceful exit from auto-battlebots
        end_ep_early = False
        if self.num_steps_taken() % 10 == 0:
            print('verify continue? set end_ep_early=True to gracefully fail out')
            ForkedPdb().set_trace()

        self._last_action_str = action_str
        action_dict = {"action": action_str}
        signal.alarm(8) # second timeout - catch missed server connection. THIS IS NOT THREAD SAFE
        try:
            self.env.step(action_dict)
        except:
            print('Controller call took too long, continue to try again or set end_ep_early to fail out instead')
            ForkedPdb().set_trace()
            self.env.step(action_dict)
        
        signal.alarm(0)

        if action_str == "Done" or end_ep_early:
            self._took_end_action = True
            dt_total = (datetime.now() - self.start_time).total_seconds()/60
            print('I think I found a ', self.task_info['object_type'], ' after ', str(dt_total), ' minutes.' )
            print('Was I correct? Set self._success in trace. Default false.')            
            ForkedPdb().set_trace()

        if self.last_time is not None:
            dt = (datetime.now() - self.last_time).total_seconds()
            print('FPS: ', str(1/dt))
        self.last_time = datetime.now()

        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        self.visualize(last_action_name)

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
        return reward
    
    def metrics(self):
        return {}



