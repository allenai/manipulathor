import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from allenact.base_abstractions.callbacks import Callback
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

from allenact.base_abstractions.sensor import Sensor
import gym
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from utils.stretch_utils.stretch_object_nav_tasks import ObjectNavTask

from manipulathor_utils.debugger_util import ForkedPdb
import cv2

class LocalLoggingSensor(Sensor[StretchManipulaTHOREnvironment,ObjectNavTask]):

    def get_observation(
        self, env:StretchManipulaTHOREnvironment, task:ObjectNavTask, *args: Any, **kwargs: Any
    ) -> Any:
        if not task.additional_visualize:
            return None

        #     draw_points(self.real_prev_location, 'g'); draw_points(self.belief_prev_location, 'b'); plt.show()
        #     ForkedPdb().set_trace()
    
        # NOTE: Create top-down trajectory path visualization
        agent_path = [
            dict(x=p["x"], y=0.25, z=p["z"])
            for p in task._metrics["task_info"]["followed_path"]
        ]
        if task.distance_type != "real_world":
            # THIS ASSUMES BOTH CAMERAS ARE ON (slash only works for stretch with one third-party camera)
            if len(env.controller.last_event.third_party_camera_frames) < 2:
                event = env.step({"action": "GetMapViewCameraProperties"})
                cam = event.metadata["actionReturn"].copy()
                cam["orthographicSize"] += 1
                env.step(
                    {"action": "AddThirdPartyCamera", "skyboxColor":"white", **cam}
                )
            event = env.step({"action": "VisualizePath", "positions":agent_path})
            env.step({"action":"HideVisualizedPath"})
            path = event.third_party_camera_frames[1]
        else:

            fig, ax = plt.subplots()
            ax = plt.axes()
            xs = [p["x"] for p in agent_path]
            zs = [p["z"] for p in agent_path]
            ax.plot(xs, zs, marker='o', color='g')
            ax.set_title("Nominal agent path from origin/start")

        df = pd.read_csv(
            f"experiment_output/ac-data/{task.task_info['id']}.txt",
            names=list(task.class_action_names())+["EstimatedValue"],
        )
        try:
            ep_length = task._metrics["ep_length"]
        except:
            ForkedPdb().set_trace()

        # get returns from each step
        returns = []
        for r in reversed(task.task_info["rewards"]):
            if len(returns) == 0:
                returns.append(r)
            else:
                returns.append(r + returns[-1] * 0.99) # gamma value
        returns = returns[::-1]

        video_frames = []
        for step in range(task._metrics["ep_length"] + 1):
            is_first_frame = step == 0
            is_last_frame = step == task._metrics["ep_length"]

            agent_frame = np.array(
                Image.fromarray(task.observations[step])#.resize((224*2, 224))
            )
            frame_number = step
            dist_to_target = task.task_info["dist_to_target"][step]

            if is_first_frame:
                last_action_success = None
                last_reward = None
                return_value = None
            else:
                last_action_success = task.task_info["action_successes"][step - 1]
                last_reward = task.task_info["rewards"][step - 1]
                return_value = returns[step - 1]

            if is_last_frame:
                action_dist = None
                critic_value = None
                taken_action = None
            else:
                policy_critic_value = df.iloc[step].values.tolist()
                action_dist = policy_critic_value[:5] # set programmatically
                critic_value = policy_critic_value[5]

                taken_action = task.task_info["taken_actions"][step]

            video_frame = LocalLogging.get_video_frame(
                agent_frame=agent_frame,
                frame_number=frame_number,
                action_names=task.class_action_names(),
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

        os.makedirs(f"experiment_output/trajectories/{task.task_info['id']}", exist_ok=True)

        imsn = ImageSequenceClip([frame for frame in video_frames], fps=10)
        imsn.write_videofile(f"experiment_output/trajectories/{task.task_info['id']}/frames.mp4")

        # save the top-down path
        if task.distance_type != "real_world":
            Image.fromarray(path).save(f"experiment_output/trajectories/{task.task_info['id']}/path.png")
        else:
            fig.savefig(f"experiment_output/trajectories/{task.task_info['id']}/path.png")
            path=np.array(Image.open(f"experiment_output/trajectories/{task.task_info['id']}/path.png")) # this is really dumb

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
            f"experiment_output/trajectories/{task.task_info['id']}/value_fn.svg",
            bbox_inches="tight",
        )

        with open(f"experiment_output/trajectories/{task.task_info['id']}/data.json", "w") as f:
            json.dump(
                {
                    "id": task.task_info["id"],
                    "spl": task._metrics["spl"],
                    "success": task._metrics["success"],
                    "finalDistance": task.task_info["dist_to_target"][-1],
                    "initialDistance": task.task_info["dist_to_target"][0],
                    "minDistance": min(task.task_info["dist_to_target"]),
                    "episodeLength": task._metrics["ep_length"],
                    "confidence": (
                        None
                        if task.task_info["taken_actions"][-1] != "End"
                        else df.End.to_list()[-1]
                    ),
                    "failedActions": len(
                        [s for s in task.task_info["action_successes"] if not s]
                    ),
                    "targetObjectType": task.task_info["object_type"],
                    "numTargetObjects": len(task.task_info["target_object_ids"]),
                    "mirrored": task.task_info["mirrored"],
                    "scene": {
                        "name": task.task_info["house_name"],
                        "split": "train",
                        "rooms": 1,
                    },
                },
                f,
            )

        return {
            "observations": task.observations,
            "path": [],#path,
            **task._metrics,
        }


class LocalLogging(Callback):
    def __init__(self):
        # NOTE: Makes it more statistically meaningful
        self.aggregate_by_means_across_n_runs: int = 10
        self.by_means_iter: int = 0
        self.by_metrics = dict()
    
    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        """Determines the data returned to the `tasks_data` parameter in the
        above *_log functions."""
        return [LocalLoggingSensor(uuid="local_logging_callback_sensor",
                                   observation_space=gym.spaces.Discrete(1)),]

    @staticmethod
    def get_columns(task: Dict[str, Any]) -> List[str]:
        """Get the columns of the quantitative table."""
        types = int, float, str, bool, wandb.Image, wandb.Video

        columns = []
        for key in task.keys():
            if isinstance(task[key], types):
                columns.append(key)

        for key in task["task_info"]:
            if isinstance(task["task_info"][key], types):
                columns.append(f"task_info/{key}")
        return columns

    @staticmethod
    def get_quantitative_table(tasks_data: List[Any], step: int) -> wandb.Table:
        """Get the quantitative table."""
        if len(tasks_data) == 0:
            return wandb.Table()

        data = []
        columns = LocalLogging.get_columns(tasks_data[0])
        columns.insert(0, "step")
        columns.insert(0, "path")
        columns.insert(0, "observations")

        for task in tasks_data:
            frames = task["observations"]
            frames_with_progress = []

            # NOTE: add progress bars
            for i, frame in enumerate(frames):
                # NOTE: flip the images if the task is mirrored
                if "mirrored" in task["task_info"] and task["task_info"]["mirrored"]:
                    frame = np.fliplr(frame)
                BORDER_SIZE = 15

                frame_with_progress = np.full(
                    (
                        frame.shape[0] + 50 + BORDER_SIZE * 2,
                        frame.shape[1] + BORDER_SIZE * 2,
                        frame.shape[2],
                    ),
                    fill_value=255,
                    dtype=np.uint8,
                )

                # NOTE: add border for action failures
                if i > 1 and not task["task_info"]["action_successes"][i - 1]:
                    frame_with_progress[0 : BORDER_SIZE * 2 + frame.shape[0]] = (
                        255,
                        0,
                        0,
                    )

                # NOTE: add the agent image
                frame_with_progress[
                    BORDER_SIZE : BORDER_SIZE + frame.shape[0],
                    BORDER_SIZE : BORDER_SIZE + frame.shape[1],
                ] = frame

                # NOTE: add the progress bar
                progress_bar = frame_with_progress[-35:-15, BORDER_SIZE:-BORDER_SIZE]
                progress_bar[:] = (225, 225, 225)
                if len(frames) > 1:
                    num_progress_pixels = int(
                        progress_bar.shape[1] * i / (len(frames) - 1)
                    )
                    progress_bar[:, :num_progress_pixels] = (38, 94, 212)

                frames_with_progress.append(frame_with_progress)

            frames = np.stack(frames_with_progress, axis=0)
            frames = np.moveaxis(frames, [1, 2, 3], [2, 3, 1])
            trajectory = wandb.Video(frames, fps=5, format="mp4")

            entry = []
            for column in columns:
                if column == "observations":
                    entry.append(trajectory)
                elif column == "step":
                    entry.append(step)
                elif column == "path":
                    entry.append(wandb.Image(task["path"]))
                elif column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    entry.append(task[column])

            data.append(entry)

        # clean up column names
        columns = [
            c[len("task_info/") :] if c.startswith("task_info/") else c for c in columns
        ]

        return wandb.Table(data=data, columns=columns)

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        **kwargs,
    ) -> None:
        """Log the train metrics to wandb."""
        table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        quantitative_table = (
            {f"train-quantitative-examples/{step:012}": table} if table.data else {}
        )

        for episode in metrics:
            by_rooms_key = (
                f"train-metrics-by-rooms/{episode['task_info']['rooms']}-rooms"
            )
            by_obj_type_key = (
                f"train-metrics-by-obj-type/{episode['task_info']['object_type']}"
            )

            for k in (by_rooms_key, by_obj_type_key):
                if k not in self.by_metrics:
                    self.by_metrics[k] = {
                        "means": {
                            "reward": 0,
                            "ep_length": 0,
                            "success": 0,
                            "spl": 0,
                            "dist_to_target": 0,
                        },
                        "count": 0,
                    }
                self.by_metrics[k]["count"] += 1
                for metric in self.by_metrics[k]["means"]:
                    old_mean = self.by_metrics[k]["means"][metric]
                    self.by_metrics[k]["means"][metric] = (
                        old_mean
                        + (episode[metric] - old_mean) / self.by_metrics[k]["count"]
                    )

        by_means_dict = {}
        self.by_means_iter += 1
        if self.by_means_iter % self.aggregate_by_means_across_n_runs == 0:
            # NOTE: log by means
            for metric, info in self.by_metrics.items():
                for mean_key, mean in info["means"].items():
                    key = f"/{mean_key}-".join(metric.split("/"))
                    by_means_dict[key] = mean
            # NOTE: reset the means
            self.by_metrics = dict()

    @staticmethod
    def get_metrics_table(tasks: List[Any]) -> wandb.Table:
        """Get the metrics table."""
        columns = LocalLogging.get_columns(tasks[0])
        data = []
        for task in tasks:
            entry = []
            for column in columns:
                if column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    entry.append(task[column])
            data.append(entry)

        columns = [
            c[len("task_info/") :] if c.startswith("task_info/") else c for c in columns
        ]
        return wandb.Table(data=data, columns=columns)

    @staticmethod
    def get_metric_plots(
        metrics: Dict[str, Any], split: Literal["valid", "test"], step: int
    ) -> Dict[str, Any]:
        """Get the metric plots."""
        plots = {}
        table = LocalLogging.get_metrics_table(metrics["tasks"])

        # NOTE: Log difficulty SPL and success rate
        if "difficulty" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "success",
                title=f"{split} Success by Difficulty ({step:,} steps)",
            )
            plots[f"{split}-spl-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "spl",
                title=f"{split} SPL by Difficulty ({step:,} steps)",
            )

        # NOTE: Log object type SPL and success rate
        if "object_type" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "success",
                title=f"{split} Success by Object Type ({step:,} steps)",
            )
            plots[f"{split}-spl-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "spl",
                title=f"{split} SPL by Object Type ({step:,} steps)",
            )

        return plots

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Log the validation metrics to wandb."""
        plots = (
            self.get_metric_plots(metrics=metrics, split="valid", step=step)
            if metrics
            else {}
        )
        table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        val_table = (
            {f"valid-quantitative-examples/{step:012}": table} if table.data else {}
        )

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Log the test metrics to wandb."""
        trajectories = []
        for filename in glob.glob("trajectories/*/data.json"):
            with open(filename, "r") as f:
                trajectories.append(json.load(f))
            os.remove(filename)

        with open("trajectories/data.json", "w") as f:
            json.dump(trajectories, f, indent=2, sort_keys=True)

    @staticmethod
    def get_video_frame(
        agent_frame: np.ndarray,
        frame_number: int,
        action_names: List[str],
        last_reward: Optional[float],
        critic_value: Optional[float],
        return_value: Optional[float],
        dist_to_target: float,
        action_dist: Optional[List[float]],
        ep_length: int,
        last_action_success: Optional[bool],
        taken_action: Optional[str],
    ) -> np.array:
        
        agent_height, agent_width, ch = agent_frame.shape

        font_to_use = "Arial.ttf" # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 14)

        IMAGE_BORDER = 25
        TEXT_OFFSET_H = 60
        TEXT_OFFSET_V = 30

        image_dims = (agent_height + 2*IMAGE_BORDER +  30,
                        agent_width + 2*IMAGE_BORDER + 200,
                        ch)
        image = np.full(image_dims, 255, dtype=np.uint8)

        
        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height, IMAGE_BORDER : IMAGE_BORDER + agent_width, :
        ] = agent_frame

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
        # font size 25, aligned center and middle
        if action_dist is not None:
            for i, (prob, action) in enumerate(
                zip(
                    action_dist,action_names
                )
            ):
                img_draw.text(
                    (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, (TEXT_OFFSET_V+5) + i * 20),
                    action,
                    font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                    fill="gray" if action != taken_action else "black",
                    anchor="rm",
                )
                img_draw.rectangle(
                    (
                        IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H+5),
                        TEXT_OFFSET_V + i * 20,
                        IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H+5) + int(100 * prob),
                        (TEXT_OFFSET_V+10) + i * 20,
                    ),
                    outline="blue",
                    fill="blue",
                )

        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,#ImageFont.truetype(font_to_use, 25),
            fill="white",
        )

        oset = -10
        if last_reward is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                "Last Reward:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                " " + ("+" if last_reward > 0 else "") + str(last_reward),
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        oset = 10
        if critic_value is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                "Critic Value:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 175 + oset),
                " " + ("+" if critic_value > 0 else "") + str(critic_value),
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        if return_value is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 195 + oset),
                "Return:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 195 + oset),
                " " + ("+" if return_value > 0 else "") + str(return_value),
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        if last_action_success is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 235),
                "Last Action:",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 235),
                " Success" if last_action_success else " Failure",
                font=full_font_load,#ImageFont.truetype(font_to_use, 14),
                fill="green" if last_action_success else "red",
                anchor="lm",
            )

        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            "Target Dist:",
            font=full_font_load,#ImageFont.truetype(font_to_use, 14),
            fill="gray",
            anchor="rm",
        )
        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            f" {dist_to_target}m",
            font=full_font_load,#ImageFont.truetype(font_to_use, 14),
            fill="gray",
            anchor="lm",
        )

        lower_offset = 10
        progress_bar_height = 20

        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + agent_width,
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="lightgray",
            fill="lightgray",
        )
        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + int(frame_number * agent_width / ep_length),
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="blue",
            fill="blue",
        )

        return np.array(text_image)
