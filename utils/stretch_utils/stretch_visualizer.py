import os
from datetime import datetime

import cv2
from torch.distributions.utils import lazy_property

from ithor_arm.ithor_arm_constants import reset_environment_and_additional_commands
from ithor_arm.ithor_arm_viz import save_image_list_to_gif, LoggerVisualizer, put_action_on_image, put_additional_text_on_image
import numpy as np

from manipulathor_utils.debugger_util import ForkedPdb


class StretchBringObjImageVisualizer(LoggerVisualizer):
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
        addition_texts = [str(x) for x in episode_info.agent_body_dist_to_obj]
        self.log_queue = put_additional_text_on_image(self.log_queue, addition_texts)
        concat_all_images = np.expand_dims(np.stack(self.arm_frame_queue, axis=0), axis=1)
        arm_frames = np.expand_dims(np.stack(self.log_queue, axis=0), axis=1)
        concat_all_images = np.concatenate([concat_all_images, arm_frames], axis=3)
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

        # self.log_start_goal(
        #     environment,
        #     task_info["visualization_source"],
        #     tag="start",
        #     img_adr=os.path.join(self.log_dir, time_to_write),
        #     additional_observations=additional_observation_start,
        #     episode_info=episode_info
        # )
        # self.log_start_goal(
        #     environment,
        #     task_info["visualization_target"],
        #     tag="goal",
        #     img_adr=os.path.join(self.log_dir, time_to_write),
        #     additional_observations=additional_observation_goal,
        #     episode_info=episode_info
        # )

        self.log_queue = []
        self.action_queue = []
        self.arm_frame_queue = []

    def log(self, environment, action_str):
        image_tensor = environment.current_frame
        arm_frame = environment.arm_frame
        self.action_queue.append(action_str)
        self.log_queue.append(image_tensor)
        self.arm_frame_queue.append(arm_frame)


    @lazy_property
    def arm_frame_queue(self):
        return []

    def log_start_goal(self, env, task_info, tag, img_adr, additional_observations=[], episode_info=None):
        object_location = task_info["object_location"]
        object_id = task_info["object_id"]
        agent_state = task_info["agent_pose"]
        this_controller = env.controller
        scene = this_controller.last_event.metadata[
            "sceneName"
        ]  # maybe we need to reset env actually]
        #We should not reset here
        # for start arm from high up as a cheating, this block is very important. never remov

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