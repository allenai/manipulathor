import copy
import math

import torch

from utils.noise_from_habitat import ControllerNoiseModel, MotionNoiseModel, _TruncatedMultivariateGaussian
import numpy as np

class NoiseInMotion:
    noise_mode = ControllerNoiseModel(
        linear_motion=MotionNoiseModel(
            _TruncatedMultivariateGaussian([0.074, 0.036], [0.019, 0.033]),
            _TruncatedMultivariateGaussian([0.189], [0.038]),
        ),
        rotational_motion=MotionNoiseModel(
            _TruncatedMultivariateGaussian([0.002, 0.003], [0.0, 0.002]),
            _TruncatedMultivariateGaussian([0.219], [0.019]),
        ),
    )

def tensor_from_dict(pos_dict):
    array = []
    if 'x' in pos_dict:
        array = [pos_dict[k] for k in ['x','y','z']]
    else:
        if 'position' in pos_dict:
            array += [pos_dict['position'][k] for k in ['x','y','z']]
        if 'rotation' in pos_dict:
            array += [pos_dict['rotation'][k] for k in ['x','y','z']]
    return torch.Tensor(array)
def squeeze_bool_mask(mask):
    if type(mask) == np.ndarray:
        mask = mask.astype(bool).squeeze(-1)
    elif type(mask) == torch.Tensor:
        mask = mask.bool().squeeze(-1)
    return mask

def add_translation_noise(change_in_xyz, prev_location, current_real_rotation, current_belief_rotation):
    if np.abs(change_in_xyz).sum() > 0:
        noise_value_x, noise_value_z = self.noise_mode.linear_motion.linear.sample() * 0.01 * self.noise #to convert to meters #TODO ?
        new_change_in_xyz = change_in_xyz.copy()
        new_change_in_xyz[0] += noise_value_x
        new_change_in_xyz[2] += noise_value_z
        # real_rotation = self.real_prev_location['camera_rotation']
        # belief_rotation = self.belief_prev_location['camera_rotation']
        diff_in_rotation = math.radians(current_belief_rotation - current_real_rotation)
        # 𝑥2=cos𝛽𝑥1−sin𝛽𝑦1
        # 𝑦2=sin𝛽𝑥1+cos𝛽𝑦1
        new_location = prev_location.copy()
        x = math.cos(diff_in_rotation) * new_change_in_xyz[0] - math.sin(diff_in_rotation) * new_change_in_xyz[2]
        z = math.sin(diff_in_rotation) * new_change_in_xyz[0] + math.cos(diff_in_rotation) * new_change_in_xyz[2]
        new_location[0] += x
        new_location[2] += z
    else:
        new_location = prev_location + change_in_xyz
    return new_location

def add_rotation_noise(change_in_rotation, prev_rotation):
    new_rotation = prev_rotation + change_in_rotation
    if change_in_rotation > 0:
        noise_in_rotation = self.noise_mode.rotational_motion.rotation.sample().item() * self.noise
        new_rotation += noise_in_rotation
    return new_rotation

def get_accurate_locations(env):
    metadata = copy.deepcopy(env.controller.last_event.metadata)
    camera_xyz = np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]])
    camera_rotation=metadata["agent"]["rotation"]["y"]
    camera_horizon=metadata["agent"]["cameraHorizon"]
    arm_state = env.get_absolute_hand_state()

    return dict(camera_xyz=camera_xyz, camera_rotation=camera_rotation, camera_horizon=camera_horizon, arm_state=arm_state)