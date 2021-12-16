import copy
import time

import torch

import pdb

# from utils.possible_object_location_utils import get_object_info, get_object_location
from scripts.dataset_generation.util_find_object_possible_location import get_object_info, get_object_location

DIST_THR = 0.1
MOVING_THR = 0.01
BASIC_HORIZON = 10

def position_distance(a,b):
    return sum([(a[k] - b[k]) ** 2 for k in ['x', 'y', 'z']]) ** 0.5

def wait_until_no_object_move(controller, object_id):
    initial_location = controller.last_event.get_object(object_id)['position']

    # make sure object is not moving
    object_moving = True
    MAX_STEPS = 30
    # while(object_moving):

    while(object_moving and MAX_STEPS >= 0): # the reason why i did this was because after first action it would say not moving but it is fake
        MAX_STEPS -= 1
        controller.step('Pass')
        object = [o for o in controller.last_event.metadata['objects'] if o['objectId'] == object_id][0]
        object_moving = object['isMoving']
        time.sleep(0.1)

    if position_distance(object['position'], initial_location) > MOVING_THR:
        return False
    else:
        return True

def prune_countertops(all_countertops):
    for counter in all_countertops:
        possible_positions = counter['possible_positions']
        pruned = []
        for position in possible_positions:
            isvalid = True
            for prev_position in pruned:
                distance = position_distance(prev_position, position)
                if distance < DIST_THR:
                    isvalid = False
                    break
            if not isvalid:
                continue
            pruned.append(position)
        counter['possible_positions'] = pruned

    return all_countertops

def find_agent_location_to_counter_visible(controller, countertop_id, reachable_positions, min_indices, horizon=BASIC_HORIZON):
    for j in range(20):
        point_index = min_indices[j]
        closest_location_to_counter = reachable_positions[point_index]
        closest_location_to_counter['horizon'] = horizon
        for i in range(0,360, 90):
            closest_location_to_counter['rotation'] = i
            pose = closest_location_to_counter
            event = controller.step(action='TeleportFull', standing=True, x=pose['x'], y=pose['y'], z=pose['z'], rotation=dict(x=0.0, y=pose['rotation'], z=0.0), horizon=pose['horizon'])
            target_obj = [o for o in event.metadata['objects'] if o['objectId'] == countertop_id and o['visible']]
            if len(target_obj) > 0:
                break
        if len(target_obj) > 0:
            break
    if len(target_obj) > 0:
        return pose
    return None

def two_position_subtracted(goal, start):
    return dict(position={
        k: goal['position'][k] - start['position'][k]
        for k in ['x', 'y', 'z']
    }, rotation={'x':0, 'y':0, 'z':0})


def transport_agent_to_closest_object(controller, object_id, reachable_positions, countertop_id):
    # make sure object is stable
    success_wait = wait_until_no_object_move(controller, object_id)

    if not success_wait:
        print('Oh no it is MOVED TOO MUCH')
        return False, {}
        # pdb.set_trace()

    reaching_object = get_object_location(controller, object_id)

    all_distances = [position_distance(k, reaching_object['position']) for k in reachable_positions]
    all_distances = torch.Tensor(all_distances)
    values, min_indices = torch.topk(all_distances, len(all_distances), largest=False)
    for j in range(20):
        point_index = min_indices[j]
        closest_location_to_counter = reachable_positions[point_index]
        closest_location_to_counter['horizon'] = BASIC_HORIZON
        for i in range(0,360, 90): # should this be 45?
            closest_location_to_counter['rotation'] = i
            pose = closest_location_to_counter
            event = controller.step(action='TeleportFull', x=pose['x'], y=pose['y'], z=pose['z'], rotation=dict(x=0.0, y=pose['rotation'], z=0.0), horizon=pose['horizon'], standing=True)
            target_obj = [o for o in event.metadata['objects'] if o['objectId'] == object_id and o['visible']]
            if len(target_obj) > 0:
                break


        if len(target_obj) > 0:
            break
    if len(target_obj) == 0:

        #Sometimes this is none
        real_countertop_id = get_object_info(controller, object_id)['parentReceptacles']
        # print('real receptacle is', real_countertop_id)

        # find a location that the countertop is visible
        visible_pose = find_agent_location_to_counter_visible(controller, countertop_id, reachable_positions, min_indices)
        if visible_pose is None:
            point_index = min_indices[0]
            visible_pose = reachable_positions[point_index]
            # print('object not visible and countertop not visible')
        event = controller.step(action='TeleportFull', x=visible_pose['x'], y=visible_pose['y'], z=visible_pose['z'], rotation=dict(x=0.0, y=visible_pose['rotation'], z=0.0), horizon=visible_pose['horizon'], standing=True)

        return False, controller.last_event.metadata['agent']
    else:
        return True, controller.last_event.metadata['agent']