import glob
import json
import os
from typing import Any, Dict, List, Optional
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from allenact.base_abstractions.callbacks import Callback
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont


class LocalLogging(Callback):
    def __init__(self):
        # NOTE: Makes it more statistically meaningful
        self.aggregate_by_means_across_n_runs: int = 10
        self.by_means_iter: int = 0
        self.by_metrics = dict()

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
        last_reward: Optional[float],
        critic_value: Optional[float],
        return_value: Optional[float],
        dist_to_target: float,
        action_dist: Optional[List[float]],
        ep_length: int,
        last_action_success: Optional[bool],
        taken_action: Optional[str],
    ) -> np.array:
        image = np.full((300, 470, 3), 255, dtype=np.uint8)

        TOP_OFFSET = 25
        image[
            TOP_OFFSET : TOP_OFFSET + 224, TOP_OFFSET : TOP_OFFSET + 224, :
        ] = agent_frame

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
        # font size 25, aligned center and middle
        if action_dist is not None:
            for i, (prob, action) in enumerate(
                zip(
                    action_dist,
                    [
                        "MoveAhead",
                        "RotateLeft",
                        "RotateRight",
                        "End",
                        "LookUp",
                        "LookDown",
                    ],
                )
            ):
                img_draw.text(
                    (TOP_OFFSET * 2 + 224 + 60, 35 + i * 20),
                    action,
                    font=ImageFont.truetype("arial.ttf", 14),
                    fill="gray" if action != taken_action else "black",
                    anchor="rm",
                )
                img_draw.rectangle(
                    (
                        TOP_OFFSET * 2 + 224 + 65,
                        30 + i * 20,
                        TOP_OFFSET * 2 + 224 + 65 + int(100 * prob),
                        40 + i * 20,
                    ),
                    outline="blue",
                    fill="blue",
                )

        img_draw.text(
            (TOP_OFFSET * 1.1, TOP_OFFSET * 1),
            str(frame_number),
            font=ImageFont.truetype("arial.ttf", 25),
            fill="white",
        )

        oset = -10
        if last_reward is not None:
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 175 + oset),
                "Last Reward:",
                font=ImageFont.truetype("arial.ttf", 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 175 + oset),
                " " + ("+" if last_reward > 0 else "") + str(last_reward),
                font=ImageFont.truetype("arial.ttf", 14),
                fill="gray",
                anchor="lm",
            )

        oset = 10
        if critic_value is not None:
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 175 + oset),
                "Critic Value:",
                font=ImageFont.truetype("arial.ttf", 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 175 + oset),
                " " + ("+" if critic_value > 0 else "") + str(critic_value),
                font=ImageFont.truetype("arial.ttf", 14),
                fill="gray",
                anchor="lm",
            )

        if return_value is not None:
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 195 + oset),
                "Return:",
                font=ImageFont.truetype("arial.ttf", 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 195 + oset),
                " " + ("+" if return_value > 0 else "") + str(return_value),
                font=ImageFont.truetype("arial.ttf", 14),
                fill="gray",
                anchor="lm",
            )

        if last_action_success is not None:
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 235),
                "Last Action:",
                font=ImageFont.truetype("arial.ttf", 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 235),
                " Success" if last_action_success else " Failure",
                font=ImageFont.truetype("arial.ttf", 14),
                fill="green" if last_action_success else "red",
                anchor="lm",
            )

        img_draw.text(
            (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 145),
            "Target Dist:",
            font=ImageFont.truetype("arial.ttf", 14),
            fill="gray",
            anchor="rm",
        )
        img_draw.text(
            (TOP_OFFSET * 2 + 224 + 60, TOP_OFFSET * 1 + 145),
            f" {dist_to_target}m",
            font=ImageFont.truetype("arial.ttf", 14),
            fill="gray",
            anchor="lm",
        )

        img_draw.rectangle(
            (
                TOP_OFFSET,
                224 + TOP_OFFSET + 10,
                TOP_OFFSET + 224,
                224 + TOP_OFFSET + 20 + 10,
            ),
            outline="lightgray",
            fill="lightgray",
        )
        img_draw.rectangle(
            (
                TOP_OFFSET,
                224 + TOP_OFFSET + 10,
                TOP_OFFSET + int(frame_number * 224 / ep_length),
                224 + TOP_OFFSET + 20 + 10,
            ),
            outline="blue",
            fill="blue",
        )

        return np.array(text_image)
