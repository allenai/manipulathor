import torch
from utils.calculation_utils import get_mid_point_of_object_from_depth_and_mask, calc_world_coordinates
from utils.noise_in_motion_util import squeeze_bool_mask
import numpy as np

from ithor_arm.arm_calculation_utils import convert_world_to_agent_coordinate


def pointnav_update(
    depth,
    image_mask,
    sequence_mask,
    prev_estimate,
    pose_update,
    camera_info,
    timestep,
    batch_index,
    agent_pose,
    name
) -> torch.FloatTensor:
    # Check if the episode reset
    if sequence_mask == 0:
        # print("reset")
        return torch.ones_like(prev_estimate) * 4.0

    # When not inited, don't update the dummy pose
    if image_mask.sum() == 0 and torch.all(prev_estimate > 3.99):
        return prev_estimate

    # Transform prev estimate to be in current agent frame
    # cos_of_pose = torch.cos(torch.deg2rad(pose_update[:, -1]))
    # sin_of_pose = torch.sin(torch.deg2rad(pose_update[:, -1]))
    # prev_estimate[:, 0] = cos_of_pose * prev_estimate[:, 0] + sin_of_pose * prev_estimate[:, 2]
    # prev_estimate[:, 2] = -sin_of_pose * prev_estimate[:, 0] + cos_of_pose* prev_estimate[:, 2]

    # prev_estimate += pose_update[:, :3]

    # If the object is not seen, keep using the prev estimate
    if image_mask.sum() == 0:
        # print("no object")
        return prev_estimate
    # print("saw", name)
    # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
    # Get current observation of the object pose in the agent frame
    estimate = get_mid_point_of_object_from_depth_and_mask(image_mask.reshape(224, 224), 
                                                           depth.reshape(224, 224),
                                                           np.zeros(3),
                                                           camera_info['xyz'][timestep][batch_index].reshape(3),
                                                           camera_info['rotation'][timestep][batch_index].reshape(1),
                                                        #    camera_info['xyz_offset'][timestep].reshape(3),
                                                        #    camera_info['rotation_offset'][timestep].reshape(1),
                                                           camera_info['horizon'][timestep][batch_index],
                                                           camera_info['fov'][timestep][batch_index],
                                                           depth.device)

    # mask = squeeze_bool_mask(image_mask).reshape(224, 224)
    # depth_frame = depth.reshape(224, 224)
    # depth_frame[~mask] = -1
    # depth_frame[depth_frame == 0] = -1
    # world_space_point_cloud = calc_world_coordinates(np.zeros(3), 
    #                                                  camera_info['xyz_offset'][timestep].reshape(3), 
    #                                                  camera_info['rotation_offset'][timestep].reshape(1), 
    #                                                  camera_info['horizon'][timestep], 
    #                                                  camera_info['fov'][timestep], 
    #                                                  depth.device, 
    #                                                  depth_frame)
    # valid_points = (world_space_point_cloud == world_space_point_cloud).sum(dim=-1) == 3
    # point_in_world = world_space_point_cloud[valid_points]
    # middle_of_object = point_in_world.mean(dim=0)
    # print("estimate", estimate, middle_of_object, estimate-middle_of_object)

    

    # If the prev_estimate is uninitialized, use the current estimate
    if torch.all(prev_estimate > 3.99):
        # print("initing")
        return estimate

    # Update the running average
    new_estimate = (prev_estimate + estimate) / 2.0
    return new_estimate
