import ast
import glob
import json

from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS
import math
import pdb
import platform

import ai2thor
import copy
import time
import random
import ai2thor.controller
import datetime
import cv2
import os
import matplotlib.pyplot as plt
import os
import numpy as np
# from utils.mid_level_constants import  scene_start_cheating_init_pose
from pyquaternion import Quaternion

from scripts.jupyter_helper import get_reachable_positions
from scripts.stretch_jupyter_helper import two_dict_equal, ARM_MOVE_CONSTANT, get_current_arm_state, only_reset_scene, transport_wrapper, ADITIONAL_ARM_ARGS, execute_command, WRIST_ROTATION, get_current_wrist_state, reset_environment_and_additional_commands

screen_size=224

STRETCH_ENV_ARGS['width'] = screen_size
STRETCH_ENV_ARGS['height'] = screen_size
STRETCH_ENV_ARGS['agentMode']='stretch'
STRETCH_ENV_ARGS['commit_id']='7184aa455bc21cc38406487a3d8e2d65ceba2571'
STRETCH_ENV_ARGS['renderDepthImage'] = True

controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)

directory = 'experiment_output/logging_debugging/'
all_files = [f for f in glob.glob(os.path.join(directory, '*.txt'))]
sorted(all_files)
all_files.reverse()
# file_name = '/Users/kianae/Desktop/2021_12_13_10_56_50_508580.txt'
def execute(file_name, verbose=False):
    with open(file_name) as f:
        lines = [l.replace('\n', '') for l in f]

    print('start executing', file_name)
    for l in lines:

        if 'reset scene' in l:
            print('executing', l)
            reset_environment_and_additional_commands(controller, l.replace('reset scene', ''))
        else:
            action_dict = ast.literal_eval(l)
            controller.step(**action_dict)
        if verbose:
            print(l)
    print('Done with No problem', file_name)

# for f in all_files:
#     execute(f)

# execute('/Users/kianae/Desktop/2021_12_13_22_00_41_298747.txt')

execute('/Users/kianae/Desktop/crashing_sample.txt', verbose=True)
pdb.set_trace()