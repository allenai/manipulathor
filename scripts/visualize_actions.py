

import argparse
import matplotlib
import matplotlib.pyplot as plt

import random
import numpy as np
import ai2thor
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from utils.stretch_utils.stretch_constants import (
    ADITIONAL_ARM_ARGS,
    PICKUP, DONE, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, MOVE_BACK, MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, MOVE_WRIST_P, MOVE_WRIST_M, MOVE_WRIST_P_SMALL, MOVE_WRIST_M_SMALL, ROTATE_LEFT_SMALL, ROTATE_RIGHT_SMALL,
)
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, STRETCH_MANIPULATHOR_COMMIT_ID, INTEL_CAMERA_WIDTH

from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS, KITCHEN_TRAIN, LIVING_ROOM_TRAIN, \
    BEDROOM_TRAIN, ROBOTHOR_TRAIN, ROBOTHOR_VAL, BATHROOM_TEST, BATHROOM_TRAIN, BEDROOM_TEST, LIVING_ROOM_TEST, \
    KITCHEN_TEST

def visualize_actions(args):

    ENV_ARGS = STRETCH_ENV_ARGS
    ENV_ARGS['commit_id'] = STRETCH_MANIPULATHOR_COMMIT_ID
    ENV_ARGS['motion_noise_type'] = 'habitat'
    ENV_ARGS['motion_noise_args'] = dict()
    ENV_ARGS['motion_noise_args']['multiplier_means'] = [1,1,1,1,1,1]
    ENV_ARGS['motion_noise_args']['multiplier_sigmas'] = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    ENV_ARGS['motion_noise_args']['effect_scale'] = .25
    ENV_ARGS['motion_noise_args']['effect_scale'] = .25
    print("ENV_ARGS", ENV_ARGS)
    TRAIN_SCENES =  KITCHEN_TRAIN + LIVING_ROOM_TRAIN + BEDROOM_TRAIN + BATHROOM_TRAIN

    env = StretchManipulaTHOREnvironment(env_args=ENV_ARGS)
    print("env", env)
    env.reset(TRAIN_SCENES[1])
    print("env", env)

    nominal_pos = []
    gt_pos = []

    nominal_pos.append([env.nominal_agent_location['x'], env.nominal_agent_location['z']])
    gt_pos.append([env.controller.last_event.metadata['agent']['position']['x'], env.controller.last_event.metadata['agent']['position']['z']])
    actions = ["MoveAhead",
               "RotateLeft",
               "RotateRight",
               "MoveAhead",
               "RotateLeft",
               "RotateLeft",
               "RotateLeft",
               "RotateLeft",
               "MoveAhead",
               "MoveAhead",
               "MoveAhead",
               "MoveAhead",
               "MoveAhead",
               "RotateLeft",
               "RotateLeft",
               "MoveAhead",
               "MoveAhead",
               "MoveAhead",
               "RotateLeft",
               "MoveAhead",
               "RotateLeft",
               "MoveAhead",
               "RotateRight",
               "RotateLeft",
               "MoveAhead"]

    for i in range(25):
        #action_dict = {'action': random.choice([MOVE_AHEAD, MOVE_AHEAD, MOVE_AHEAD, ROTATE_RIGHT, ROTATE_LEFT, MOVE_BACK])}
        action_dict = {'action': actions[i]}
        print("action_dict", action_dict['action'])
        env.step(action_dict)
        nominal_pos.append([env.nominal_agent_location['x'], env.nominal_agent_location['z']])
        gt_pos.append([env.controller.last_event.metadata['agent']['position']['x'], env.controller.last_event.metadata['agent']['position']['z']])
#        print("nominal", env.nominal_agent_location)
#        print("sim", env.controller.last_event.metadata['agent']['position'])

    nominal_pos = np.array(nominal_pos)
    gt_pos = np.array(gt_pos)

    matplotlib.use('Qt5Agg')
    plt.plot(nominal_pos[:, 0], nominal_pos[:, 1], 'r', label="Nominal")
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'g', label="Ground Truth")
    plt.plot([gt_pos[0, 0], nominal_pos[0, 0]], [gt_pos[0, 1], nominal_pos[0, 1]], 'bo', label="Starting location")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=16)
    args = parser.parse_args()

    visualize_actions(args)
