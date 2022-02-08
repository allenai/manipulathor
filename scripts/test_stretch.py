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
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, STRETCH_MANIPULATHOR_COMMIT_ID

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





action_options = ['m', 'r', 'l', 'b', 'hu', 'hd', 'ao', 'ai', 'go', 'gc', 'wp', 'wn']

kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

all_scenes = kitchens + living_rooms + bedrooms + bathrooms
NUM_TESTS = 100
EPS_LEN = 500

def setup_thirdparty_camera(controller, camera_position):
    # controller.step('Pass')
    if len(controller.last_event.third_party_camera_frames) > 1:
        controller.step('UpdateThirdPartyCamera',
            thirdPartyCameraId=1, # id is available in the metadata response
            rotation=camera_position['rotation'],
            position=camera_position['position']
            )
    else:
        controller.step('AddThirdPartyCamera', 
            rotation=camera_position['rotation'], 
            position=camera_position['position'],
            fieldOfView=100)

def visualize(controller, save=False, addition_str=''):
    camera_setup = 2
    if camera_setup == 1:
        agent_rotation = {'x':0, 'y':225, 'z':0}
        agent_camera = controller.last_event.metadata['cameraPosition']
        agent_camera['y'] += 0.2
        agent_camera['z'] += 1
        agent_camera['x'] += 1
        camera_position = {
            'position': agent_camera,
            'rotation':
                agent_rotation
        }
    if camera_setup == 2:
        agent_rotation = {'x':90, 'y':0, 'z':0}
        agent_camera = controller.last_event.metadata['cameraPosition']
        agent_camera['y'] += 0.5
        agent_camera['z'] += 0
        agent_camera['x'] += 0
        camera_position = {
            'position': agent_camera,
            'rotation':
                agent_rotation
        }

    # controller.step('Pass')
    setup_thirdparty_camera(controller, camera_position)
    image = controller.last_event.frame
    arm_view = controller.last_event.third_party_camera_frames[0]
    third_view = controller.last_event.third_party_camera_frames[1]

    combined=np.concatenate([image, arm_view, third_view], 0)
    
    imagename = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S.%f")
    imagename += addition_str
    if save:
        plt.imsave(os.path.join(saved_image_folder, imagename+'.png'), combined)
    else:
        plt.cla()
        plt.imshow(combined)
        plt.show()

def manual_task(controller, scene_name, logger_number =0, final=False, save_frames = False, init_sequence=[], verbose = False):
    only_reset_scene(controller, scene_name)
    all_actions = []
    all_action_details = []
    actions_ran_so_far = 0
    while(True):
        visualize(controller, save_frames, addition_str='action_{}'.format('s' if controller.last_event.metadata['lastActionSuccess'] else 'f'))
        if len(init_sequence) > 0:
            action = init_sequence[0]
            init_sequence = init_sequence[1:]
        else:
            action = input('action?')
        if action == 'q':
            break
        all_actions.append(action)

        detail = execute_command(controller, action, ADITIONAL_ARM_ARGS)
        if verbose:
            print(detail, controller.last_event.metadata['lastActionSuccess'])
        all_action_details.append(detail)
        actions_ran_so_far += 1

        # controller.step(action='MoveArm', coordinateSpace="wrist", position=dict(x=0, y=0.2,z=0.2));visualize(controller, save_frames)
        # controller.step(action='RotateWristRelative', yaw=90);visualize(controller, save_frames)

    print(scene_name)
    print(all_actions)
    print(all_action_details)
    pdb.set_trace()

def print_locations(controller):
    location = copy.deepcopy(controller.last_event.metadata['arm']['joints'])
#     print(controller.last_event)
    print([x['rootRelativePosition']['z'] for x in location])
#     print([x['rootRelativeRotation']['y'] for x in location])
    visualize(controller, save=True)


def test_arm_movements(controller, scenes= all_scenes, num_tests=NUM_TESTS, episode_len=EPS_LEN, visualize_tests=False, one_by_one=False):
    #TODO add p and d
    ALL_POSSIBLE_ACTIONS = ['hu', 'hd', 'ao', 'ai'] + ['m', 'r', 'l', 'b'] + ['wp', 'wn']
    times = [1]
    for i in range(num_tests):
        if one_by_one:
            scene = scenes[i % len(scenes)]
        else:
            scene = random.choice(scenes)
        try:
            controller.reset(scene)
        except Exception:
            print('Failed for scene, ', scene)
            controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)#, renderInstanceSegmentation=True)
            controller.reset(scene)

        print('fps', 1.0 / (sum(times) / len(times)))
        print('Test', scene, i)
        all_seq = []
        all_action_succ = []
        detailed_actions = []
        free_motion = -1
        for seq_i in range(episode_len):
            if free_motion==-1:
                action = random.choice(ALL_POSSIBLE_ACTIONS)
            else:
                action = ALL_POSSIBLE_ACTIONS[free_motion]
                free_motion+=1
            #


            all_seq.append(action)
            def get_all_states(controller):
                arm = get_current_arm_state(controller)
                wrist = get_current_wrist_state(controller)
                agent = controller.last_event.metadata['agent']
                return arm, wrist, agent
            arm_before_action, wrist_before_action, agent_before_action = get_all_states(controller)
            before = datetime.datetime.now()
            list_of_action = execute_command(controller, action, ADITIONAL_ARM_ARGS)
            after = datetime.datetime.now()
            times.append((after-before).microseconds * 1e-6)
            detailed_actions.append(list_of_action)
            all_action_succ.append(controller.last_event.metadata['lastActionSuccess'])
            arm_after_action, wrist_after_action, agent_after_action = get_all_states(controller)
            expected_arm_after_action = copy.deepcopy(arm_before_action)
            expected_wrist_after_action = copy.deepcopy(wrist_before_action)
            if controller.last_event.metadata['lastActionSuccess']:
                if action in ['m', 'b']: #TODO this is not super accurate but just for now
                    distances = [agent_before_action['position'][k] - agent_after_action['position'][k] for k in ['x', 'y', 'z']]
                    sum_distances = sum([abs(k) for k in distances])
                    if sum_distances < 0.05:
                        print('ERROR in Agent Move', scene, all_seq, detailed_actions)
                        pdb.set_trace()
                        break
                if action in ['r', 'l']:
                    distances = [agent_before_action['rotation'][k] - agent_after_action['rotation'][k] for k in ['x', 'y', 'z']]
                    sum_distances = sum([abs(k) for k in distances])
                    if sum_distances < 10:
                        print('ERROR in Agent Rotation', scene, all_seq, detailed_actions)
                        pdb.set_trace()
                        break
                if action in ['hu', 'hd', 'ao', 'ai']:
                    if action == 'hu':
                        expected_arm_after_action['y'] += ARM_MOVE_CONSTANT
                    if action == 'hd':
                        expected_arm_after_action['y'] -= ARM_MOVE_CONSTANT
                    if action == 'ao':
                        expected_arm_after_action['z'] += ARM_MOVE_CONSTANT
                    if action == 'ai':
                        expected_arm_after_action['z'] -= ARM_MOVE_CONSTANT
                    if expected_arm_after_action['z'] <= 0 or expected_arm_after_action['y'] > 1.04 or expected_arm_after_action['z'] > 0.8 or expected_arm_after_action['y'] <= -0.05:
                        pass
                    elif not two_dict_equal(expected_arm_after_action, arm_after_action):
                        print('ERROR in ARM', scene, all_seq, detailed_actions)
                        pdb.set_trace()
                        break
                if action in ['wp', 'wn']: #
                    wrist_dist = (wrist_before_action * wrist_after_action.inverse).radians
                    if abs(abs(math.degrees(wrist_dist)) - WRIST_ROTATION) > 1e-2:
                        print('ERROR in WRIST', scene, all_seq, detailed_actions)
                        pdb.set_trace()
                        break
                free_motion=-1
            else:
                if free_motion == -1:
                    free_motion = 0
                    random.shuffle(ALL_POSSIBLE_ACTIONS)
                else:
                    if free_motion == len(ALL_POSSIBLE_ACTIONS):
                        print('STUCK', scene, all_seq, detailed_actions)
                        break
            if visualize_tests:
                visualize(controller, True)
        if seq_i == episode_len - 1:
            print('Test Passed', len([x for x in all_action_succ if x == True]), 'out of', len(all_action_succ))
        # else:
        #     pdb.set_trace()

def test_arm_scene_generalizations(controller):
    print('test arm openning all scenes')
    for scene in all_scenes:
        try:
            controller.reset(scene)
        except Exception:
            print('Failed to Start', scene)
            controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)
    print('finished test arm openning all scenes')

def test_teleport_agent(controller, scenes=all_scenes):
    for scene in scenes:
        controller.reset(scene)
        reachable_positions = get_reachable_positions(controller)
        failed = 0
        for position in reachable_positions:
            rotation = {'x':0, 'y':random.choice([i * 30 for i in range(120)]), 'z':0}
            teleport_detail = dict(action='TeleportFull', standing=True, x=position['x'], y=position['y'], z=position['z'], rotation=dict(x=rotation['x'], y=rotation['y'], z=rotation['z']), horizon=10)
            event_TeleportFull = controller.step(teleport_detail)
            if event_TeleportFull:
                if not two_dict_equal(controller.last_event.metadata['agent']['position'], position):
                    print('Failed to teleport but said successful')
            else:
                failed += 1
                # print(event_TeleportFull)
                # print('scene', scene)
                # print('step', teleport_detail)
        print('scene', scene, 'failed', failed, 'out of', len(reachable_positions))

def test_fov(controller):
    STRETCH_ENV_ARGS['width'] = int(720/3)
    STRETCH_ENV_ARGS['height'] = int(1280/3)
    # STRETCH_ENV_ARGS['agentMode'] = 'arm'
    print(STRETCH_ENV_ARGS)
    controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)
    print('Done')
    manual_task(controller, 'FloorPlan2', logger_number =0, final=False, save_frames=True)


# In[26]:

if __name__ == '__main__':

    controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)#, renderInstanceSegmentation=True)

    #TODO add pickup and drop tests

    # # all the following tests need to pass
    print('Test 1')
    test_arm_scene_generalizations(controller)

    print('Test 2')
    print('Testing arm stuck in all scenes')
    test_arm_movements(controller, scenes=all_scenes, num_tests=len(all_scenes), episode_len = 30, visualize_tests=False, one_by_one=True)
    print('Finished Testing arm stuck in all scenes')
    #
    print('Test 3')
    print('Random tests')
    test_arm_movements(controller, scenes=all_scenes, num_tests=1000, visualize_tests=False)
    # test_arm_movements(controller, scenes=all_scenes, num_tests=1000, visualize_tests=True)
    print('Finished')

    print('Test 4')
    test_teleport_agent(controller)
    test_fov(controller)


    # manual_task(controller, 'FloorPlan2', logger_number =0, final=False, save_frames=True)

    # manual_task('FloorPlan15', logger_number =0, final=False, save_frames=True, init_sequence=['m', 'r', 'hd', 'wp', 'ai', 'm', 'hu', 'b', 'wn', 'l', 'ao'], verbose = True)
    # manual_task('FloorPlan15', logger_number =0, final=False, save_frames=True, init_sequence=[], verbose = True)
    # manual_task('FloorPlan4', logger_number =0, final=False, save_frames=True, init_sequence=['wp', 'm', 'hd', 'hu', 'wp', 'ai', 'r', 'ao', 'l', 'wn', 'b'])

    # test_arm_movements(controller, scenes=['FloorPlan2'], num_tests=100, visualize_tests=True)
    # controller.reset('FloorPlan2')
    # controller.step(action='MoveArm', position=dict(x=0, y=1, z=1),)
    # print_locations(controller)
    # controller.step(action='RotateWristRelative', yaw=90)
    # print_locations(controller)
# before=get_current_wrist_state(controller);before; controller.step(action='RotateWristRelative', yaw=10);visualize(controller, save_frames); after=get_current_wrist_state(controller);after; Quaternion.sym_distance(before, after)

