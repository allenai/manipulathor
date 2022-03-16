import torch
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import depth_frame_to_world_space_xyz
import numpy as np

def position_distance(s1, s2):
    position1 = s1["position"]
    position2 = s2["position"]
    return (
        (position1["x"] - position2["x"]) ** 2
        + (position1["y"] - position2["y"]) ** 2
        + (position1["z"] - position2["z"]) ** 2
    ) ** 0.5


def calc_world_coordinates(min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, device, depth_frame):
    with torch.no_grad():
        camera_xyz = (
            torch.from_numpy(camera_xyz - min_xyz).float().to(device)
        )

        depth_frame = torch.from_numpy(depth_frame).to(device)
        depth_frame[depth_frame == -1] = np.NaN
        world_space_point_cloud = depth_frame_to_world_space_xyz(
            depth_frame=depth_frame,
            camera_world_xyz=camera_xyz,
            rotation=camera_rotation,
            horizon=camera_horizon,
            fov=fov,
        )
        return world_space_point_cloud