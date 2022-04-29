import cv2
import cv2.aruco as aruco
import numpy as np

from manipulathor_utils.debugger_util import ForkedPdb
from utils.calculation_utils import calc_world_coordinates
from utils.noise_in_motion_util import squeeze_bool_mask

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_detection_parameters =  aruco.DetectorParameters_create()
# Apparently available in OpenCV 3.4.1, but not OpenCV 3.2.0.
aruco_detection_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
aruco_detection_parameters.cornerRefinementWinSize = 2

def test_aruco_marker_detection(cv2_image):
    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    aruco_corners, aruco_ids, aruco_rejected_image_points = aruco.detectMarkers(gray_image,aruco_dict,parameters = aruco_detection_parameters)
    return aruco_corners

def get_binary_mask_of_arm(rgb_image):
    rgb_image = rgb_image.copy()
    bgr_image = rgb_image[:,:,::-1]
    valid_corners = test_aruco_marker_detection(bgr_image)
    arm_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
    if len(valid_corners) > 1:
        print('Multiple Valid corners')
        ForkedPdb().set_trace()
    elif len(valid_corners) == 0:
        print('Arm not detected')
        # ForkedPdb().set_trace()
        pass #no valid is found
    elif len(valid_corners) == 1:
        corners = valid_corners[0].squeeze(0).astype(int)
        start_point = corners.min(axis=0)
        end_point = corners.max(axis=0)
        arm_mask = cv2.rectangle(arm_mask, start_point, end_point, (1,1,1), -1)
    return arm_mask

def get_mid_point_of_object_from_depth_and_mask(mask, depth_frame_original, min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, device):
    mask = squeeze_bool_mask(mask)
    depth_frame_masked = depth_frame_original.copy()
    depth_frame_masked[~mask] = -1
    depth_frame_masked[depth_frame_masked == 0] = -1 # This means they are either not existing or not valid
    world_space_point_cloud = calc_world_coordinates(min_xyz, camera_xyz, camera_rotation, camera_horizon, fov, device, depth_frame_masked)
    valid_points = (world_space_point_cloud == world_space_point_cloud).sum(dim=-1) == 3
    point_in_world = world_space_point_cloud[valid_points]
    midpoint_agent_coord = point_in_world.mean(dim=0)
    return midpoint_agent_coord