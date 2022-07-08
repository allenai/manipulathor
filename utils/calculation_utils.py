import torch
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import depth_frame_to_world_space_xyz, camera_space_xyz_to_world_xyz
from utils.noise_in_motion_util import squeeze_bool_mask
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
        if isinstance(camera_xyz, np.ndarray):
            camera_xyz = (
                torch.from_numpy(camera_xyz - min_xyz).float().to(device)
            )
        else:
            camera_xyz = camera_xyz - torch.from_numpy(min_xyz).float().to(device)
        if isinstance(depth_frame, np.ndarray):
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

def calc_world_xyz_from_agent_relative(object_xyz_list, agent_xyz, agent_rotation, device):
    # This might just be a duplication/convenience thing of the above
    with torch.no_grad():
        agent_xyz = (torch.from_numpy(agent_xyz).float().to(device))

        object_xyz_matrix = np.ndarray([3,len(object_xyz_list)])
        for i in range(len(object_xyz_list)):
            x,y,z = object_xyz_list[i]['position']['x'],object_xyz_list[i]['position']['y'],object_xyz_list[i]['position']['z']
            object_xyz_matrix[:,i] = [x,y,z]
        object_xyz_matrix = torch.from_numpy(object_xyz_matrix).to(device)

        world_xyz_matrix = camera_space_xyz_to_world_xyz(object_xyz_matrix, agent_xyz, agent_rotation, horizon=0 ).cpu().numpy()
        world_positions = []
        for i in range(len(object_xyz_list)):
            x,y,z = world_xyz_matrix[:,i]
            world_positions.append(dict(position=dict(x=x, y=y,z=z), rotation=dict(x=0,y=0, z=0)))

        return world_positions

def get_mid_point_of_object_from_depth_and_mask(mask, depth_frame_original, min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, device):
    mask = squeeze_bool_mask(mask)
    if isinstance(depth_frame_original, np.ndarray):
        depth_frame_masked = depth_frame_original.copy()
    else:
        depth_frame_masked = depth_frame_original.clone()
    depth_frame_masked[~mask] = -1
    depth_frame_masked[depth_frame_masked == 0] = -1 # TODO: what is this for? missing values?
    world_space_point_cloud = calc_world_coordinates(min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, device, depth_frame_masked)
    valid_points = (world_space_point_cloud == world_space_point_cloud).sum(dim=-1) == 3
    point_in_world = world_space_point_cloud[valid_points]
    midpoint_agent_coord = point_in_world.mean(dim=0)
    return midpoint_agent_coord
