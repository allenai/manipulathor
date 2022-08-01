import torch

from manipulathor_baselines.stretch_bring_object_baselines.experiments.ithor.pointnav_emul_stretch_all_rooms_noisy_motion_with_isdf \
    import PointNavEmulStretchAllRoomsNoisyMotionWithISDF


class PointNavEmulStretchAllRoomsNoisyMotionWithISDFDistrib(
    PointNavEmulStretchAllRoomsNoisyMotionWithISDF
):
    NUM_PROCESSES = 8
    def __init__(
            self,
            distributed_nodes: int = 3,
    ):
        super().__init__()
        self.distributed_nodes = distributed_nodes
        self.train_gpu_ids = tuple(range(torch.cuda.device_count())) # should I do this for everyone?, should i add val

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