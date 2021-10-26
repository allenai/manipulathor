import platform

import gym
import torch
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from torch import nn

from ithor_arm.bring_object_sensors import CategorySampleSensor, NoisyObjectMask, NoGripperRGBSensorThor
from ithor_arm.bring_object_task_samplers import DiverseBringObjectTaskSampler
from ithor_arm.bring_object_tasks import WPickUPExploreBringObjectTask, ExploreWiseRewardTask
from ithor_arm.ithor_arm_constants import ENV_ARGS, TRAIN_OBJECTS, TEST_OBJECTS
from ithor_arm.ithor_arm_sensors import (
    InitialAgentArmToObjectSensor,
    InitialObjectToGoalSensor,
    PickedUpObjSensor,
    DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor,
)
from ithor_arm.ithor_arm_viz import MaskImageVisualizer
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_ddppo import BringObjectMixInPPOConfig
from manipulathor_baselines.bring_object_baselines.experiments.bring_object_mixin_simplegru import BringObjectMixInSimpleGRUConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import BringObjectiThorBaseConfig
from manipulathor_baselines.bring_object_baselines.experiments.ithor.complex_reward_no_pu import ComplexRewardNoPU
from manipulathor_baselines.bring_object_baselines.experiments.ithor.complex_reward_no_pu_w_memory import ComplexRewardNoPUWMemory
from manipulathor_baselines.bring_object_baselines.experiments.ithor.complex_reward_with_pickup import ComplexRewardWPickupExp
from manipulathor_baselines.bring_object_baselines.models.gt_mask_with_memory_model import MemoryWGtMaskRGBDModel
from manipulathor_baselines.bring_object_baselines.models.query_obj_w_gt_mask_rgb_model import SmallBringObjectWQueryObjGtMaskRGBDModel



class ComplexRewardWPickupExpDistrib(
    ComplexRewardWPickupExp
):
    NUM_PROCESSES = 30
    def __init__(
            self,
            distributed_nodes: int = 1,
    ):
        super().__init__()
        self.distributed_nodes = distributed_nodes
        self.train_gpu_ids = tuple(range(torch.cuda.device_count())) #TODO should I do this for everyone?, should i add val


    def machine_params(self, mode="train", **kwargs):
        params = super().machine_params(mode, **kwargs)

        if mode == "train":
            params.devices = params.devices * self.distributed_nodes
            params.nprocesses = params.nprocesses * self.distributed_nodes
            params.sampler_devices = params.sampler_devices * self.distributed_nodes

            if "machine_id" in kwargs:
                machine_id = kwargs["machine_id"]
                assert (
                        0 <= machine_id < self.distributed_nodes
                ), f"machine_id {machine_id} out of range [0, {self.distributed_nodes - 1}]"

                local_worker_ids = list(
                    range(
                        len(self.train_gpu_ids) * machine_id,
                        len(self.train_gpu_ids) * (machine_id + 1),
                        )
                )

                params.set_local_worker_ids(local_worker_ids)

            # Confirm we're setting up train params nicely:
            print(
                f"devices {params.devices}"
                f"\nnprocesses {params.nprocesses}"
                f"\nsampler_devices {params.sampler_devices}"
                f"\nlocal_worker_ids {params.local_worker_ids}"
            )
        elif mode == "valid":
            # Use all GPUs at their maximum capacity for training
            # (you may run validation in a separate machine)
            params.nprocesses = (0,)

        return params