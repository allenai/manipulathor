import argparse
import pdb

import ai2thor.controller
import ai2thor
import json
import sys, os

from ithor_arm.arm_calculation_utils import initialize_arm
from ithor_arm.ithor_arm_constants import TRAIN_OBJECTS, reset_environment_and_additional_commands, transport_wrapper, TEST_OBJECTS, ENV_ARGS
from scripts.jupyter_helper import is_object_in_receptacle, is_agent_at_position

sys.path.append(os.path.abspath('.'))
# from plugins.ithor_arm_plugin.ithor_arm_constants import reset_environment_and_additional_commands, TRAIN_OBJECTS, TEST_OBJECTS, ENV_ARGS, transport_wrapper
#
# from plugins.ithor_arm_plugin.arm_calculation_utils import initialize_arm, is_object_in_receptacle, is_agent_at_position

SCENES = ["FloorPlan{}_physics".format(str(i + 1)) for i in range(30)]

OBJECTS_TO_WORK = TRAIN_OBJECTS # + TEST_OBJECTS#
# OBJECTS_TO_WORK = TEST_OBJECTS

SOURCE_VERSION = ""
PRUNED_VERSION = "pruned_v3_"

assert SOURCE_VERSION != PRUNED_VERSION

def test_initial_location(controller):
    for s in SCENES:
        reset_environment_and_additional_commands(controller, s)
        event1, event2, event3 = initialize_arm(controller)
        if not (event1.metadata['lastActionSuccess'] and event2.metadata['lastActionSuccess'] and event3.metadata['lastActionSuccess']):
            return False, 'failed for {}'.format(s)

    return True, ''

def check_datapoint_correctness(controller, source_location):
    scene = source_location['scene_name']
    reset_environment_and_additional_commands(controller, scene)
    _1, _2, _3 = initialize_arm(controller) #This is checked before
    event_place_obj = transport_wrapper(controller, source_location['object_id'], source_location['object_location'])
    # _1, _2, _3 = initialize_arm(controller) #This is checked before
    agent_state = source_location['agent_pose']
    teleport_detail = dict(action='TeleportFull', standing=True, x=agent_state['position']['x'], y=agent_state['position']['y'], z=agent_state['position']['z'], rotation=dict(x=agent_state['rotation']['x'], y=agent_state['rotation']['y'], z=agent_state['rotation']['z']), horizon=agent_state['cameraHorizon'])
    event_TeleportFull = controller.step(teleport_detail)

    object_id = source_location['object_id']
    object_state = [o for o in event_TeleportFull.metadata['objects'] if o['objectId'] == object_id][0]
    object_is_visible = object_state['visible']
    object_correct_location = is_object_in_receptacle(controller.last_event,object_id,source_location['countertop_id'])
    agent_correct_location = is_agent_at_position(controller, teleport_detail)
    # check to transport object

    # check to do arm init
    # check to transport agent
    # check object is visible
    if event_place_obj.metadata['lastActionSuccess'] and event_TeleportFull.metadata['lastActionSuccess'] and object_is_visible and object_correct_location and agent_correct_location:
        return True, ''
    else:
        error_reasons = []
        if not event_place_obj.metadata['lastActionSuccess']:
            error_reasons.append('transport')
        if not event_TeleportFull.metadata['lastActionSuccess']:
            error_reasons.append('teleport')
        if not object_is_visible:
            error_reasons.append('visible')
        if not object_correct_location:
            error_reasons.append('object_correct_location')
        if not agent_correct_location:
            error_reasons.append('agent_correct_location')
        return False, error_reasons
        # return False, 'Data point invalid for {}, because of event_place_obj {}, event_TeleportFull{}, object_is_visible {}'.format(source_location, event_place_obj, event_TeleportFull, object_is_visible)

def test_train_data_points(controller, object_names):
    total_checked = 0
    total_error = 0
    total_reasons = {}
    for s in SCENES:
        for o in object_names:
            print('Testing ', s, o)

            with open('datasets/apnd-dataset/valid_object_positions/{}valid_{}_positions_in_{}.json'.format(SOURCE_VERSION, o, s)) as f:
                data_points = json.load(f)
            visible_data = [data for data in data_points[s] if data['visibility']]
            for datapoint in visible_data:
                result, message = check_datapoint_correctness(controller, datapoint)
                total_checked += 1
                if not result:
                    total_error += 1
                    for reason in message:

                        total_reasons.setdefault(reason, 0)
                        total_reasons[reason] += 1
                    # return False, message
                    print('total checked', total_checked, 'total error', total_error, 'reasons', total_reasons)

    return True, ''


def prune_data_points(controller, object_names):
    for s in SCENES:
        for o in object_names:
            print('Pruning ', s, o)
            pruned_version_name = 'datasets/apnd-dataset/valid_object_positions/{}valid_{}_positions_in_{}.json'.format(PRUNED_VERSION, o, s)
            if os.path.exists(pruned_version_name):
                print('This file exists')
                pdb.set_trace()

            with open('datasets/apnd-dataset/valid_object_positions/{}valid_{}_positions_in_{}.json'.format(SOURCE_VERSION, o, s)) as f:
                data_points = json.load(f)
            visible_data = [data for data in data_points[s] if data['visibility']]
            remaining_valid = []
            for ind, datapoint in enumerate(visible_data):
                if ind % 100 == 10:
                    print(ind, 'out of', len(visible_data))
                result, message = check_datapoint_correctness(controller, datapoint)
                if result:
                    remaining_valid.append(datapoint)
            print('out of ', len(visible_data), 'remained', len(remaining_valid))

            with open(pruned_version_name, 'w') as f:
                json.dump({s: remaining_valid}, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Sync')
    parser.add_argument('--prune', default=False, action='store_true')
    parser.add_argument('--test_objects', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()


    if args.test_objects:
        object_names = TEST_OBJECTS
    else:
        object_names = OBJECTS_TO_WORK

    controller = ai2thor.controller.Controller(
        **ENV_ARGS
        # gridSize=0.25,
        # width=224, height=224, agentMode='arm', fieldOfView=100,
        # agentControllerType='mid-level',
        # server_class=ai2thor.fifo_server.FifoServer, visibilityScheme='Distance',
        # useMassThreshold = True, massThreshold = 10,
        # visibilityDistance = 1.25
    )

    print('Testing initial location')
    result, message = test_initial_location(controller)
    if result:
        print('Passed')
    else:
        print('Failed', message)

    if args.prune:
        print('Are you sure?')
        prune_data_points(controller, object_names)
    else:

        print('Testing test_train_data_points')
        result, message = test_train_data_points(controller, object_names)
        if result:
            print('Passed')
        else:
            print('Failed', message)