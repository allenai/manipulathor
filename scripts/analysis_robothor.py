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
from scripts.stretch_jupyter_helper import two_dict_equal, ARM_MOVE_CONSTANT, get_current_arm_state, only_reset_scene, transport_wrapper, ADITIONAL_ARM_ARGS, execute_command, WRIST_ROTATION, get_current_wrist_state
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, STRETCH_MANIPULATHOR_COMMIT_ID, ROBOTHOR_SCENE_NAMES

screen_size=224

STRETCH_ENV_ARGS['width'] = screen_size
STRETCH_ENV_ARGS['height'] = screen_size
STRETCH_ENV_ARGS['agentMode']='stretch'
# STRETCH_ENV_ARGS['commit_id']='03b26e96a43c83f955386b8cac925d4d2b550837'
STRETCH_ENV_ARGS['commit_id'] = STRETCH_MANIPULATHOR_COMMIT_ID
STRETCH_ENV_ARGS['commit_id'] = 'f698c1c27a39536858c854cae413fd31987cdf2a' #TODO jsut a test for speed segmentation
STRETCH_ENV_ARGS['renderDepthImage'] = True
STRETCH_ENV_ARGS['renderInstanceSegmentation'] = True #TODO try out some real segmentation

if platform.system() == "Darwin":
    saved_image_folder = '/Users/kianae/Desktop/saved_stretch_images'
    os.makedirs(saved_image_folder, exist_ok=True)




def test_stretch_in_robothor():
    # # all the following tests need to pass
    global STRETCH_ENV_ARGS
    STRETCH_ENV_ARGS['commit_id'] = 'fe005524939307669392dab264a22da8ab6ed53a' #TODO should we put this everywhere?
    controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)
    print('Testing ', controller._build.url)
    list_of_all_objects = {}
    exact_total_number = {}
    for scene in ROBOTHOR_SCENE_NAMES:
        controller.reset(scene)
        all_object_types = [o['objectType'] for o in controller.last_event.metadata['objects']]
        object_names = set(all_object_types)
        object_counts = {k:all_object_types.count(k) for k in object_names}
        print('scene', scene, ':', object_counts)
        for k in object_names:
            list_of_all_objects.setdefault(k, 0)
            list_of_all_objects[k] += 1
            exact_total_number.setdefault(k, 0)
            exact_total_number[k] += object_counts[k]
    print(list_of_all_objects)
    exists_in_every_room = [k for (k,v) in list_of_all_objects.items() if v == 75]
    exists_uniquely_in_every_room = [k for (k,v) in exact_total_number.items() if v == 75]
    # overlap_with_ithor = [k for k in exists_uniquely_in_every_room if k in ]
    pdb.set_trace()


if __name__ == '__main__':
    # test_stretch_in_THOR()
    test_stretch_in_robothor()


