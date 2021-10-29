"""Task Definions for the task of ArmPointNav"""

from typing import Dict, Tuple, List, Any, Optional, Union

import gym
import numpy as np
import torch
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)

from ithor_arm.bring_object_tasks import AbstractBringObjectTask, BringObjectTask
from ithor_arm.ithor_arm_constants import (
    MOVE_ARM_CONSTANT,
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_X_P,
    MOVE_ARM_X_M,
    MOVE_ARM_Y_P,
    MOVE_ARM_Y_M,
    MOVE_ARM_Z_P,
    MOVE_ARM_Z_M,
    MOVE_AHEAD,
    MOVE_LEFT,
    MOVE_RIGHT,
    ROTATE_RIGHT,
    ROTATE_LEFT,
    LOOK_UP,
    LOOK_DOWN,
    PICKUP,
    DONE, ARM_LENGTH, MOVE_BACK,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_viz import LoggerVisualizer
from manipulathor_utils.debugger_util import ForkedPdb
from scripts.jupyter_helper import get_reachable_positions


class ExploreTask(BringObjectTask):
    _actions = (
        MOVE_BACK,
        MOVE_AHEAD,
        MOVE_LEFT,
        MOVE_RIGHT,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        LOOK_UP,
        LOOK_DOWN
        #NOTE @samir add look up and look down here and also add it to the environment
    )
    def obj_distance_from_goal(self):
        return 0
    def arm_distance_from_obj(self):
        return 0
    def get_original_object_distance(self):
        return 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_locations = [[k['x'], k['y'], k['z']] for k in get_reachable_positions(self.env.controller)]
        self.all_reachable_positions = torch.Tensor(all_locations) #TODO @samir you can change this to cover more than just the lcoations and add observed objects as well
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))
        self.seen_objects = set()
        self.prop_seen_after = 0
        self.steps_taken = 0

        # NOTE: from luca
        self.visited_positions_xzrsh = {self.agent_location_tuple}
        self.visited_positions_xz = {self.agent_location_tuple[:2]}
        self.seen_pickupable_objects = set(
            o["name"] for o in self.pickupable_objects(visible_only=True)
        )
        self.seen_openable_objects = set(
            o["name"] for o in self.openable_not_pickupable_objects(visible_only=True)
        )
        self.total_pickupable_or_openable_objects = len(
            self.pickupable_or_openable_objects(visible_only=False)
        )

    # def judge(self) -> float:
    #     """Compute the reward after having taken a step."""
    #     reward = self.reward_configs["step_penalty"]

    #     ForkedPdb().set_trace()

    #     # TODO @samir add reward for objects
    #     current_agent_location = self.env.get_agent_location()
    #     current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
    #     all_distances = self.all_reachable_positions - current_agent_location
    #     all_distances = (all_distances ** 2).sum(dim=-1)
    #     location_index = torch.argmin(all_distances)
    #     if self.has_visited[location_index] == 0:
    #         reward += self.reward_configs["exploration_reward"]
    #     self.has_visited[location_index] = 1

    #     # TODO: mess with this
    #     for o in self.env.visible_objects():
    #         if o['name'] not in self.seen_objects:
    #             reward += self.reward_configs["object_reward"]
    #         self.seen_objects.add(o['name'])


    #     if not self.last_action_success:
    #         reward += self.reward_configs["failed_action_penalty"]

    #     if self._took_end_action:
    #         reward += (
    #             self.reward_configs["goal_success_reward"]
    #             if self._success
    #             else self.reward_configs["failed_stop_reward"]
    #         )

    #     # add collision cost, maybe distance to goal objective,...
    #     return float(reward)

    def judge(self) -> float:

        reward_kiana = self.reward_configs["step_penalty"]

        # TODO @samir add reward for objects
        current_agent_location = self.env.get_agent_location()
        current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['y'], current_agent_location['z']])
        all_distances = self.all_reachable_positions - current_agent_location
        all_distances = (all_distances ** 2).sum(dim=-1)
        location_index = torch.argmin(all_distances)
        if self.has_visited[location_index] == 0:
            reward_kiana += self.reward_configs["exploration_reward"]
        else:
            reward_kiana += self.reward_configs['visted_reward']
        self.has_visited[location_index] = 1

        self.steps_taken += 1

        if not self.last_action_success:
            reward_kiana += self.reward_configs["failed_action_penalty"]


        """Return the reward from a new (s, a, s'). NOTE: From Luca"""
        total_seen_before = len(self.seen_pickupable_objects) + len(
            self.seen_openable_objects
        )
        prop_seen_before = (
            total_seen_before
        ) / self.total_pickupable_or_openable_objects

        # Updating seen openable
        for obj in self.openable_not_pickupable_objects(visible_only=True):
            if obj["name"] not in self.seen_openable_objects:
                self.seen_openable_objects.add(obj["name"])

        # Updating seen pickupable
        for obj in self.pickupable_objects(visible_only=True):
            if obj["name"] not in self.seen_pickupable_objects:
                self.seen_pickupable_objects.add(obj["name"])

        # Updating visited locations
        agent_loc_tuple = self.agent_location_tuple
        self.visited_positions_xzrsh.add(agent_loc_tuple)
        if agent_loc_tuple[:2] not in self.visited_positions_xz:
            self.visited_positions_xz.add(agent_loc_tuple[:2])

        total_seen_after = len(self.seen_pickupable_objects) + len(
            self.seen_openable_objects
        )
        prop_seen_after = total_seen_after / self.total_pickupable_or_openable_objects
        self.prop_seen_after = prop_seen_after

        reward = 5 * (prop_seen_after - prop_seen_before)

        if self.steps_taken == self.max_steps and prop_seen_after > 0.5:
            reward += 5 * (prop_seen_after + (prop_seen_after > 0.98))

        return reward + reward_kiana

    def metrics(self) -> Dict[str, Any]:
        result = super(AbstractBringObjectTask, self).metrics()
        if self.is_done():
            result['percent_room_visited'] = self.has_visited.mean()
            result['prop_seen_after'] = self.prop_seen_after
            result["success"] = self._success
            # TODO @samir add metric for obect, logged in tb automatically
            self.finish_visualizer_metrics(result)
            self.finish_visualizer(self._success)
            self.action_sequence_and_success = []
        return result
    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]

        self._last_action_str = action_str
        action_dict = {"action": action_str}

        self.env.step(action_dict)
        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        last_action_success = float(self.last_action_success)
        self.action_sequence_and_success.append((last_action_name, last_action_success))
        self.visualize(last_action_name)


        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    @staticmethod
    def agent_location_to_tuple(
        agent_loc: Dict[str, Union[Dict[str, float], bool, float, int]]
    ) -> Tuple[float, float, int, int, int]:
        if "position" in agent_loc:
            agent_loc = {
                "x": agent_loc["position"]["x"],
                "y": agent_loc["position"]["y"],
                "z": agent_loc["position"]["z"],
                "rotation": agent_loc["rotation"]["y"],
                "horizon": agent_loc["cameraHorizon"],
                "standing": agent_loc.get("isStanding"),
            }
        return (
            round(agent_loc["x"], 2),
            round(agent_loc["z"], 2),
            round_to_factor(agent_loc["rotation"], 90) % 360,
            1 * agent_loc["standing"],
            round_to_factor(agent_loc["horizon"], 30) % 360,
        )

    @property
    def agent_location_tuple(self) -> Tuple[float, float, int, int, int]:
        return self.agent_location_to_tuple(self.env.get_agent_location())

    def pickupable_objects(self, visible_only: bool = True):
        with include_object_data(self.env.controller):
            return [
                o
                for o in self.env.last_event.metadata["objects"]
                if ((o["visible"] or not visible_only) and o["pickupable"])
            ]

    def openable_not_pickupable_objects(self, visible_only: bool = True):
        with include_object_data(self.env.controller):
            return [
                o
                for o in self.env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only)
                    and (o["openable"] and not o["pickupable"])
                )
            ]

    def pickupable_or_openable_objects(self, visible_only: bool = True):
        with include_object_data(self.env.controller):
            return [
                o
                for o in self.env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only)
                    and (o["pickupable"] or (o["openable"] and not o["pickupable"]))
                )
            ]