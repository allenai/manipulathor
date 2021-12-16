from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from typing import Any, Union, Optional

import gym
import numpy as np
# from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, Sensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
import cv2


from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
from ithor_arm.bring_object_sensors import NoisyObjectMask
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_sensors import DepthSensorThor
from manipulathor_utils.debugger_util import ForkedPdb


class DepthSensorStretch(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = (env.controller.last_event.depth_frame.copy())
        depth = clip_frame(depth)
        #TODO the ratio of image is slightly different in the real stretch tho
        return depth


class DepthSensorStretchIntel(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = (env.controller.last_event.depth_frame.copy())
        return depth


class DepthSensorStretchKinect(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = env.controller.last_event.third_party_depth_frames[0].copy()
        return depth


class DepthSensorStretchKinectZero(
    DepthSensorThor
):
    """Sensor for Depth images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        depth = env.controller.last_event.third_party_depth_frames[0].copy()
        depth[:] = 0
        return depth

class RGBSensorStretchKinect(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = env.controller.last_event.third_party_camera_frames[0].copy()
        return rgb


class RGBSensorStretchKinectZero(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = env.controller.last_event.third_party_camera_frames[0].copy()
        rgb[:] = 0
        return rgb

class RGBSensorStretchIntel(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = (env.controller.last_event.frame.copy())

        return rgb#cv2.resize(rgb, (224,224))

class NoisyObjectMaskStretch(NoisyObjectMask): #TODO double check correctness of this

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        mask = super().get_observation(env, task, *args, **kwargs)
        return clip_frame(mask)
class RGBSensorStretch(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:

        rgb = (env.controller.last_event.frame.copy())
        rgb = clip_frame(rgb) #TODO we should add more noise to this as well
        #TODO this is very dorehami
        return rgb
#TODO we need to crop our segmentation masks as well.
MASK_FRAMES = None

def clip_frame(frame):
    #TODO should we swap this w and h?
    if len(frame.shape) == 2:
        w, h = frame.shape
    if len(frame.shape) == 3:
        w, h, c = frame.shape
    if MASK_FRAMES is None or MASK_FRAMES.shape[0] != w or MASK_FRAMES.shape[1] != h:
        set_mask_frames(w, h)
    frame[(1 - MASK_FRAMES).astype(bool)] = 0
    return frame


def set_mask_frames(w, h):
    original_w, original_h = 640, 576
    w_up_left, h_up_left = 150, 270
    w_up_right, h_up_right = 120, 210
    w_down_left, h_down_left = 150,250
    w_down_right, h_down_right = 120, 200

    init = [(0, 0), (original_w, 0), (0, original_h), (original_w, original_h)]
    ws = [w_up_left, -w_up_right, w_down_left, -w_down_right]
    hs = [h_up_left, h_up_right, -h_down_left, -h_down_right]

    mask = np.ones((original_h, original_w))

    for i in range(4):
        pt1 = init[i]
        pt2 = (pt1[0] + ws[i], pt1[1])
        pt3 = (pt1[0], pt1[1] + hs[i])
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        mask = cv2.drawContours(mask, [triangle_cnt], 0, (0), -1)
    global MASK_FRAMES
    MASK_FRAMES = cv2.resize(mask, (h, w))