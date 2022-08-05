import copy
import glob
import json
import os
import pdb
import platform
import random
import sys
from collections import defaultdict
from datetime import datetime

import torch
from ai2thor.controller import Controller
import datasets
import pickle

from shapely.geometry import Point, Polygon
import numpy as np

from ithor_arm.ithor_arm_constants import MANIPULATHOR_ENV_ARGS
from scripts.stretch_jupyter_helper import reset_environment_and_additional_commands, make_all_objects_unbreakable, \
    ADITIONAL_ARM_ARGS
from scripts.test_stretch import visualize, manual_task
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, PROCTHOR_COMMIT_ID


def reset_procthor_scene(controller, house):
    controller.reset()
    controller.step(action="CreateHouse", house=house)
    controller.step("ResetObjectFilter")
    event_init_teleport = controller.step(action="TeleportFull", **house["metadata"]["agent"])
    controller.step(action="MakeAllObjectsMoveable")
    controller.step(action="MakeObjectsStaticKinematicMassThreshold")
    make_all_objects_unbreakable(controller)
    event_init_arm = controller.step(dict(action="MoveArm", position=dict(x=0,y=0.8,z=0), **ADITIONAL_ARM_ARGS))
    return event_init_teleport, event_init_arm

def get_rooms_polymap(house):
    room_poly_map = {}
    # NOTE: Map the rooms
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon(
            [(p["x"], p["z"]) for p in room["floorPolygon"]]
        )
    return room_poly_map

def get_reachable_positions_with_room_id(controller, room_polymap):
    rp_event = controller.step(action="GetReachablePositions")
    reachable_positions = rp_event.metadata["actionReturn"]
    reachable_positions_w_room_dict = defaultdict(list)
    agent_location_to_room_id = {}

    if reachable_positions is None:
        return None, None, None
    reachable_positions_matrix = np.array([[pos['x'], pos['y'], pos['z']] for pos in reachable_positions])

    for agent_pose in reachable_positions:
        point = Point(agent_pose["x"], agent_pose["z"])
        room_id = None
        for room_id, poly in room_polymap.items():
            if poly.contains(point):
                reachable_positions_w_room_dict[room_id].append(
                    agent_pose
                )
                agent_location_to_room_id[(agent_pose['x'], agent_pose['y'], agent_pose['z'])] = room_id
                break
        if room_id is None:
            print(agent_pose, 'is out of house')
    return reachable_positions_w_room_dict, reachable_positions_matrix, agent_location_to_room_id
def get_visible_objects_with_their_info(controller, valid_pickupable_objects, agent_location_matrix, agent_location_to_room_id):
    object_infos = {}
    agent_pose = {'horizon': 20, 'position': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'x': 0, 'y': 0, 'z': 0}, 'standing': True}
    for obj in valid_pickupable_objects:
        obj_type = obj['objectType']
        obj_loc = np.array([obj['position']['x'],obj['position']['y'],obj['position']['z']])
        object_id = obj['objectId']
        agent_distance = np.linalg.norm(agent_location_matrix - obj_loc, axis=-1)
        if len(agent_distance) < 20:
            min_indices = [i for i in range(len(agent_distance))]
        else:
            values, min_indices = torch.topk(torch.Tensor(agent_distance), 20, largest=False)
        target_obj = []
        for j in range(len(min_indices)):
            agent_location_to_consider = agent_location_matrix[min_indices[j]]
            agent_pose['position']['x'],agent_pose['position']['y'],agent_pose['position']['z'] = agent_location_to_consider
            for i in range(0,360, 90):
                agent_pose['rotation']['y'] = i
                event = controller.step(action='TeleportFull', **agent_pose)
                target_obj = [o for o in event.metadata['objects'] if o['objectId'] == object_id and o['visible']]
                if len(target_obj) > 0:
                    break
            if len(target_obj) > 0:
                break
        if len(target_obj) > 0:
            agent_pose_key = (agent_pose['position']['x'], agent_pose['position']['y'],agent_pose['position']['z'])
            if agent_pose_key not in agent_location_to_room_id:
                print('Agent pose not found', agent_pose_key)
                continue
            room_id = agent_location_to_room_id[agent_pose_key]

            object_infos[object_id] = obj
            object_infos[object_id]['agent_pose'] = agent_pose

            object_infos[object_id]['room_id'] = room_id
            if platform.system() == "Darwin":
                img_dir = 'datasets/dataset_visualization_procthor'
                os.makedirs(img_dir, exist_ok=True)
                import matplotlib.pyplot as plt
                now = datetime.now()
                time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
                plt.imsave(os.path.join(img_dir, f'{time_to_write}_{obj_type}.png'), controller.last_event.frame)
    print('number of valid objects', len(object_infos))
    return object_infos
def get_valid_rotations(controller, room_id_to_agent_locations):
    pruned_agent_poses = {}
    for room in room_id_to_agent_locations:
        for pose in room_id_to_agent_locations[room]:
            agent_pose = {'horizon': 20, 'position': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'x': 0, 'y': 0, 'z': 0}, 'standing': True}
            agent_pose['position'] = pose
            possible_rotations = [i for i in range(0, 360, 90)]
            random.shuffle(possible_rotations) #to have variety
            for rotation in possible_rotations:
                agent_pose['rotation']['y'] = rotation
                event = controller.step(action='TeleportFull', **agent_pose)
                if event.metadata['lastActionSuccess']:
                    pruned_agent_poses.setdefault(room, [])
                    pruned_agent_poses[room].append(agent_pose)
                    if platform.system() == "Darwin":
                        img_dir = '/Users/kianae/Desktop/agent_pose_images'
                        os.makedirs(img_dir, exist_ok=True)
                        import matplotlib.pyplot as plt
                        now = datetime.now()
                        time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
                        plt.imsave(os.path.join(img_dir, f'{time_to_write}_{room}.png'), controller.last_event.frame)
                    break
    return pruned_agent_poses

def initialize_environment():
    # env_to_work_with = copy.deepcopy(STRETCH_ENV_ARGS)
    # env_to_work_with['branch'] = 'nanna-stretch'
    # env_to_work_with['branch'] = 'nanna-culling-stretch'
    # env_to_work_with['branch'] = 'nanna'
    env_to_work_with = copy.deepcopy(MANIPULATHOR_ENV_ARGS) TODO back to stretch?
    env_to_work_with['scene'] = 'Procedural'
    env_to_work_with['visibilityDistance'] = 2
    env_to_work_with['commit_id'] = PROCTHOR_COMMIT_ID
    env_to_work_with['renderInstanceSegmentation'] = False

    controller = Controller(**env_to_work_with)
    return controller
def generate_dataset_for_scenes(scene_ids):
    controller = initialize_environment()
    house_dataset = datasets.load_dataset("allenai/houses", use_auth_token=True)



    for scene in scene_ids:
        house_id_to_room_to_agent_pose = {}


        dataset_dir = 'datasets/procthor_apnd_dataset/agent_poses/'
        previously_done = [f for f in glob.glob(os.path.join(dataset_dir, f'room_id_{scene}_*.json'))]
        if len(previously_done) > 0:
            print('already done with ', scene)
            continue
        print('Working on scene', scene)
        # load the house, get reachanble positions, get all object pickupbale ids, find the location near them, if visible put it in the dictionary
        # Load the house
        house_entry = house_dataset["train"][scene]
        house = pickle.loads(house_entry["house"])
        event_init_teleport, event_init_arm = reset_procthor_scene(controller, house)
        if event_init_teleport.metadata['lastActionSuccess'] is False:
            print('Failed to teleport agent for room ', scene)
            continue
        if event_init_arm.metadata['lastActionSuccess'] is False:
            print('Failed to init arm for agent for room ', scene)
            continue
        room_polymap = get_rooms_polymap(house)
        # get reachable positions room index
        room_id_to_agent_locations, reachable_positions_matrix, agent_location_to_room_id = get_reachable_positions_with_room_id(controller, room_polymap)

        if room_id_to_agent_locations is None:
            print('No reachable position for', scene)
            continue



        # record rotation for each reachable position while arm is up
        pruned_reachable_positions = get_valid_rotations(controller, room_id_to_agent_locations)


        remained_locations = [len(v) for (k,v) in pruned_reachable_positions.items()]

        print('possible agent location', len(reachable_positions_matrix), 'possible real locations', sum(remained_locations))

        house_id_to_room_to_agent_pose[scene] = pruned_reachable_positions

        os.makedirs(dataset_dir, exist_ok=True)
        now = datetime.now()
        time_to_write = now.strftime("%m_%d_%Y_%H_%M_%S_%f")

        with open(os.path.join(dataset_dir, f'room_id_{scene}_{time_to_write}.json'), 'w') as f:
            json.dump({
                'house_id_to_room_to_agent_pose': house_id_to_room_to_agent_pose,

            }, f)

if __name__ == '__main__':

    starting_ind = int(sys.argv[-1])
    generate_dataset_for_scenes([i for i in range(starting_ind,starting_ind + 1000)])
