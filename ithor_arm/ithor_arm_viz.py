"""Utility functions and classes for visualization and logging"""
import os
from datetime import datetime
from typing import Dict

import cv2
import imageio
import numpy as np
import torch
from gym.spaces import Discrete, Box

from ithor_arm.arm_calculation_utils import initialize_arm
from ithor_arm.ithor_arm_constants import (
    reset_environment_and_additional_commands,
    transport_wrapper, MOVE_AHEAD,ROTATE_LEFT ,ROTATE_RIGHT ,MOVE_ARM_HEIGHT_P ,MOVE_ARM_HEIGHT_M ,MOVE_ARM_X_P ,MOVE_ARM_X_M ,MOVE_ARM_Y_P ,MOVE_ARM_Y_M ,MOVE_ARM_Z_P ,MOVE_ARM_Z_M ,PICKUP ,DONE
)
# from manipulathor_baselines.bring_object_baselines.models.small_bring_object_pred_box_model import SmallBringObjectPredictBBXDepthBaselineActorCritic
from manipulathor_utils.debugger_util import ForkedPdb


class LoggerVisualizer:
    def __init__(self, exp_name="", log_dir=""):
        if log_dir == "":
            log_dir = self.__class__.__name__
        if exp_name == "":
            exp_name = "NoNameExp"
        now = datetime.now()
        self.exp_name = exp_name
        log_dir = os.path.join(
            "experiment_output/visualizations",
            exp_name,
            log_dir + "_" + now.strftime("%m_%d_%Y_%H_%M_%S_%f"),
        )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_queue = []
        self.action_queue = []
        self.logger_index = 0

    def log(self, environment, action_str):
        raise Exception("Not Implemented")

    def is_empty(self):
        return len(self.log_queue) == 0

    def finish_episode_metrics(self, episode_info, task_info, metric_results):
        pass

    def finish_episode(self, environment, episode_info, task_info):
        pass


class TestMetricLogger(LoggerVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_metric_dict = {}
        log_file_name = os.path.join(
            self.log_dir, "test_metric_{}.txt".format(self.exp_name)
        )
        self.metric_log_file = open(log_file_name, "w")

    def average_dict(self):
        result = {}
        for (k, v) in self.total_metric_dict.items():
            result[k] = sum(v) / len(v)
        return result

    def finish_episode_metrics(self, episode_info, task_info, metric_results=None):

        if metric_results is None:
            print("had to reset")
            self.log_queue = []
            self.action_queue = []
            return

        for k in metric_results.keys():
            if "metric" in k or k in ["ep_length", "reward", "success"]:
                self.total_metric_dict.setdefault(k, [])
                self.total_metric_dict[k].append(metric_results[k])
        print(
            "total",
            len(self.total_metric_dict["success"]),
            "average test metric",
            self.average_dict(),
        )

        # save the task info and all the action queue and results
        log_dict = {
            "task_info_metrics": metric_results,
            "action_sequence": self.action_queue,
            "logger_number": self.logger_index,
        }
        self.logger_index += 1
        self.metric_log_file.write(str(log_dict))
        self.metric_log_file.write("\n")
        print("Logging to", self.metric_log_file.name)

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):
        # We can add agent arm and state location if needed
        self.action_queue.append(action_str)
        self.log_queue.append(action_str)

class BringObjImageVisualizer(LoggerVisualizer):
    def finish_episode(self, environment, episode_info, task_info):
        now = datetime.now()
        time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        time_to_write += "log_ind_{}".format(self.logger_index)
        self.logger_index += 1
        print("Loggigng", time_to_write, "len", len(self.log_queue))

        source_object_id = task_info["source_object_id"]
        goal_object_id = task_info["goal_object_id"]
        pickup_success = episode_info.object_picked_up
        episode_success = episode_info._success

        # Put back if you want the images
        # for i, img in enumerate(self.log_queue):
        #     image_dir = os.path.join(self.log_dir, time_to_write + '_seq{}.png'.format(str(i)))
        #     cv2.imwrite(image_dir, img[:,:,[2,1,0]])

        episode_success_offset = "succ" if episode_success else "fail"
        pickup_success_offset = "succ" if pickup_success else "fail"

        gif_name = (
                time_to_write
                + "_from_"
                + source_object_id.split("|")[0]
                + "_to_"
                + goal_object_id.split("|")[0]
                + "_pickup_"
                + pickup_success_offset
                + "_episode_"
                + episode_success_offset
                + ".gif"
        )


        self.log_queue = put_action_on_image(self.log_queue, self.action_queue[1:])
        concat_all_images = np.expand_dims(np.stack(self.log_queue, axis=0), axis=1)
        save_image_list_to_gif(concat_all_images, gif_name, self.log_dir)
        this_controller = environment.controller
        scene = this_controller.last_event.metadata[
            "sceneName"
        ]
        reset_environment_and_additional_commands(this_controller, scene)

        additional_observation_start = []
        additional_observation_goal = []
        if 'target_object_mask' in episode_info.get_observations():
            additional_observation_start.append('target_object_mask')
        if 'target_location_mask' in episode_info.get_observations():
            additional_observation_goal.append('target_location_mask')

        self.log_start_goal(
            environment,
            task_info["visualization_source"],
            tag="start",
            img_adr=os.path.join(self.log_dir, time_to_write),
            additional_observations=additional_observation_start,
            episode_info=episode_info
        )
        self.log_start_goal(
            environment,
            task_info["visualization_target"],
            tag="goal",
            img_adr=os.path.join(self.log_dir, time_to_write),
            additional_observations=additional_observation_goal,
            episode_info=episode_info
        )

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):
        image_tensor = environment.current_frame
        self.action_queue.append(action_str)
        self.log_queue.append(image_tensor)


    def log_start_goal(self, env, task_info, tag, img_adr, additional_observations=[], episode_info=None):
        object_location = task_info["object_location"]
        object_id = task_info["object_id"]
        agent_state = task_info["agent_pose"]
        this_controller = env.controller
        scene = this_controller.last_event.metadata[
            "sceneName"
        ]  # maybe we need to reset env actually]
        #We should not reset here
        # for start arm from high up as a cheating, this block is very important. never remove
        event1, event2, event3 = initialize_arm(this_controller)
        if not (
                event1.metadata["lastActionSuccess"]
                and event2.metadata["lastActionSuccess"]
                and event3.metadata["lastActionSuccess"]
        ):
            print("ERROR: ARM MOVEMENT FAILED in logging! SHOULD NEVER HAPPEN")

        event = transport_wrapper(this_controller, object_id, object_location)
        if event.metadata["lastActionSuccess"] == False:
            print("ERROR: oh no could not transport in logging")

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
        if event.metadata["lastActionSuccess"] == False:
            print("ERROR: oh no could not teleport in logging")

        image_tensor = this_controller.last_event.frame
        image_dir = (
                img_adr + "_obj_" + object_id.split("|")[0] + "_pickup_" + tag + ".png"
        )
        cv2.imwrite(image_dir, image_tensor[:, :, [2, 1, 0]])

        # Saving the mask
        if len(additional_observations) > 0:
            observations = episode_info.get_observations()
            for sensor_name in additional_observations:
                assert sensor_name in observations
                mask_frame = (observations[sensor_name])
                mask_dir = (
                        img_adr + "_obj_" + object_id.split("|")[0] + "_pickup_" + tag + "_{}.png".format(sensor_name)
                )
                cv2.imwrite(mask_dir, mask_frame.astype(float)*255.)

class MaskImageVisualizer(LoggerVisualizer):
    def finish_episode(self, environment, episode_info, task_info):
        ForkedPdb().set_trace()
        now = datetime.now()
        time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        time_to_write += "log_ind_{}".format(self.logger_index)
        self.logger_index += 1
        print("Loggigng", time_to_write, "len", len(self.log_queue))
        object_id = task_info["objectId"]

        pickup_success = episode_info.object_picked_up
        episode_success = episode_info._success

        # Put back if you want the images
        # for i, img in enumerate(self.log_queue):
        #     image_dir = os.path.join(self.log_dir, time_to_write + '_seq{}.png'.format(str(i)))
        #     cv2.imwrite(image_dir, img[:,:,[2,1,0]])

        episode_success_offset = "succ" if episode_success else "fail"
        pickup_success_offset = "succ" if pickup_success else "fail"
        gif_name = (
                time_to_write
                + "_obj_"
                + object_id.split("|")[0]
                + "_pickup_"
                + pickup_success_offset
                + "_episode_"
                + episode_success_offset
                + ".gif"
        )
        concat_all_images = np.expand_dims(np.stack(self.log_queue, axis=0), axis=1)
        save_image_list_to_gif(concat_all_images, gif_name, self.log_dir)

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):
        # ForkedPdb().set_trace()
        image_tensor = environment.current_frame
        self.action_queue.append(action_str)
        self.log_queue.append(image_tensor)


class ImageVisualizer(LoggerVisualizer):
    def finish_episode(self, environment, episode_info, task_info):
        now = datetime.now()
        time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        time_to_write += "log_ind_{}".format(self.logger_index)
        self.logger_index += 1
        print("Loggigng", time_to_write, "len", len(self.log_queue))
        object_id = task_info["source_object_id"]

        pickup_success = episode_info.object_picked_up
        episode_success = episode_info._success

        # Put back if you want the images
        # for i, img in enumerate(self.log_queue):
        #     image_dir = os.path.join(self.log_dir, time_to_write + '_seq{}.png'.format(str(i)))
        #     cv2.imwrite(image_dir, img[:,:,[2,1,0]])

        episode_success_offset = "succ" if episode_success else "fail"
        pickup_success_offset = "succ" if pickup_success else "fail"
        gif_name = (
            time_to_write
            + "_obj_"
            + object_id.split("|")[0]
            + "_pickup_"
            + pickup_success_offset
            + "_episode_"
            + episode_success_offset
            + ".gif"
        )
        concat_all_images = np.expand_dims(np.stack(self.log_queue, axis=0), axis=1)
        save_image_list_to_gif(concat_all_images, gif_name, self.log_dir)

        if 'visualization_source' in task_info:
            self.log_start_goal(
                environment,
                task_info["visualization_source"],
                tag="start",
                img_adr=os.path.join(self.log_dir, time_to_write),
            )
        if 'visualization_target' in task_info:
            self.log_start_goal(
                environment,
                task_info["visualization_target"],
                tag="goal",
                img_adr=os.path.join(self.log_dir, time_to_write),
            )

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):
        image_tensor = environment.current_frame
        # if action_str in ['MoveAheadContinuous', 'RotateRightContinuous', 'RotateLeftContinuous']: TODO remove this
        self.action_queue.append(action_str)
        self.log_queue.append(image_tensor)

    def log_start_goal(self, env, task_info, tag, img_adr):
        object_location = task_info["object_location"]
        object_id = task_info["object_id"]
        agent_state = task_info["agent_pose"]
        this_controller = env.controller
        scene = this_controller.last_event.metadata[
            "sceneName"
        ]  # maybe we need to reset env actually]
        reset_environment_and_additional_commands(this_controller, scene)
        # for start arm from high up as a cheating, this block is very important. never remove
        event1, event2, event3 = initialize_arm(this_controller)
        if not (
            event1.metadata["lastActionSuccess"]
            and event2.metadata["lastActionSuccess"]
            and event3.metadata["lastActionSuccess"]
        ):
            print("ERROR: ARM MOVEMENT FAILED in logging! SHOULD NEVER HAPPEN")

        event = transport_wrapper(this_controller, object_id, object_location)
        if event.metadata["lastActionSuccess"] == False:
            print("ERROR: oh no could not transport in logging")

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
        if event.metadata["lastActionSuccess"] == False:
            print("ERROR: oh no could not teleport in logging")

        image_tensor = this_controller.last_event.frame
        image_dir = (
            img_adr + "_obj_" + object_id.split("|")[0] + "_pickup_" + tag + ".png"
        )
        cv2.imwrite(image_dir, image_tensor[:, :, [2, 1, 0]])


def save_image_list_to_gif(image_list, gif_name, gif_dir):
    gif_adr = os.path.join(gif_dir, gif_name)

    seq_len, cols, w, h, c = image_list.shape

    pallet = np.zeros((seq_len, w, h * cols, c))

    for col_ind in range(cols):
        pallet[:, :, col_ind * h : (col_ind + 1) * h, :] = image_list[:, col_ind]

    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    imageio.mimsave(gif_adr, pallet.astype(np.uint8), format="GIF", duration=1 / 5)
    print("Saved result in ", gif_adr)

def put_action_on_image(images, actions):
    all_images = []
    for i in range(len(images) - 1):
        img = images[i]
        action = actions[i]
        action_names = (MOVE_AHEAD,ROTATE_LEFT ,ROTATE_RIGHT ,MOVE_ARM_HEIGHT_P ,MOVE_ARM_HEIGHT_M ,MOVE_ARM_X_P ,MOVE_ARM_X_M ,MOVE_ARM_Y_P ,MOVE_ARM_Y_M ,MOVE_ARM_Z_P ,MOVE_ARM_Z_M ,PICKUP ,DONE)
        action_short = ("MOVE_AHEAD","ROTATE_L" ,"ROTATE_R" ,"ARM_H_P" ,"ARM_H_M" ,"ARM_X_P" ,"ARM_X_M" ,"ARM_Y_P" ,"ARM_Y_M" ,"ARM_Z_P" ,"ARM_Z_M" ,"PICKUP" ,"DONE")
        action = action_short[action_names.index(action)]
        position = (10,10)

        from PIL import Image, ImageFont, ImageDraw
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, action, (0,0,0))
        all_images.append(np.array(pil_img))


    all_images.append(images[-1]) # No action needs to be written here
    return all_images