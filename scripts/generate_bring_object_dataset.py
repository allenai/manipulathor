import ai2thor.controller
import os
import json
import pdb
import random

from ithor_arm.ithor_arm_constants import ENV_ARGS, reset_environment_and_additional_commands, transport_wrapper
from scripts.jupyter_helper import is_object_in_receptacle, get_parent_receptacles

initial_object = 'Apple'
initial_scene = 'FloorPlan1_physics'
goal_object = 'Mug'
dataset_adr = 'datasets/apnd-dataset'
controller = ai2thor.controller.Controller(**ENV_ARGS)

def read_files(data_adr, scene_name, object_name):
    file_name = os.path.join(data_adr, 'valid_{}_positions_in_{}.json'.format(object_name, scene_name))
    with open(file_name) as f:
        locations = json.load(f)
    return locations

def check_validity(controller, event, location):
    object_correct_location = is_object_in_receptacle(controller.last_event,location['object_id'],location['countertop_id'])
    # if event.metadata['lastActionSuccess'] and not object_correct_location:  remove this
    #     all_receptacles = get_parent_receptacles(controller.last_event, location['object_id'])
    #     print('ERROR:', location['object_id'], location['countertop_id'], all_receptacles)
    return event.metadata['lastActionSuccess'] and object_correct_location

def find_feasible_pairs(initial_object_locations, goal_object_locations):

    scene_name = [k for k in initial_object_locations.keys()][0]
    initial_object_locations = initial_object_locations[scene_name]
    goal_object_locations = goal_object_locations[scene_name]

    #TODO instead make sure we have samples from each counter top, also 30 is too small maybe? Or maybe set something that checks for the total number of valids to be certain amount
    random.shuffle(initial_object_locations)
    random.shuffle(goal_object_locations)
    initial_object_locations = initial_object_locations[:30]
    goal_object_locations = goal_object_locations[:30]

    total_possible = len(initial_object_locations) * len(goal_object_locations)
    print('Calculating', scene_name, total_possible)
    so_far = 0
    possible_pairs = []
    for init_location in initial_object_locations:
        for goal_location in goal_object_locations:
            so_far += 1
            if so_far % 100 == 0:
                print(so_far, '/', total_possible)
            reset_environment_and_additional_commands(controller, scene_name)
            event1 = transport_wrapper(controller, init_location['object_id'], init_location['object_location'])
            event2 = transport_wrapper(controller, goal_location['object_id'], goal_location['object_location'])
            # we need to check object location as well? same thing we used for sanity check?
            if check_validity(controller, event1, init_location) and check_validity(controller, event2, goal_location):
                possible_pairs.append(dict(scene_name=scene_name,init_location=init_location, goal_location=goal_location))
    return possible_pairs


def find_all_feasible_pairs(initial_object, scene_name, goal_object, dataset_adr):
    valid_object_position_adr = os.path.join(dataset_adr, 'valid_object_positions')

    # read initial object locations
    initial_object_locations = read_files(valid_object_position_adr, scene_name, initial_object)
    goal_object_locations = read_files(valid_object_position_adr, scene_name, goal_object)

    # for each pair try whether it is feasible and they both succeed or not
    possible_pairs = find_feasible_pairs(initial_object_locations, goal_object_locations)

    # save them in a file
    goal_object_pair_adr = os.path.join(dataset_adr, 'valid_object_pairs')
    os.makedirs(goal_object_pair_adr, exist_ok=True)
    output_file_name = os.path.join(goal_object_pair_adr, 'valid_{}_to_{}_in_{}.json'.format(initial_object, goal_object, scene_name))
    print('saving', len(possible_pairs), 'in', output_file_name)
    with open(output_file_name, 'w') as f:
        json.dump({
            'locations': possible_pairs
        }, f)


find_all_feasible_pairs(initial_object, initial_scene, goal_object, dataset_adr)
#TODO calculate for all scenes and objects

