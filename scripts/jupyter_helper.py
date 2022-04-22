import copy
import math
import random
import ai2thor
import pdb
import ai2thor.fifo_server


### CONSTANTS
# from scripts.utils.ithor_arm_constants import MOVE_THR
# from ithor_arm.ithor_arm_constants import MOVE_THR
from manipulathor_utils.debugger_util import ForkedPdb

OBJECT_MOVEMENT_THR = 0.01

ADITIONAL_ARM_ARGS = {
    'disableRendering': True,
    'returnToStart': True,
    'speed': 1,
}

ARM_MOVE_CONSTANT = 0.05

SCENE_INDICES = [i + 1 for i in range(30)] +[i + 1 for i in range(200,230)] +[i + 1 for i in range(300,330)] +[i + 1 for i in range(400,430)]
SCENE_NAMES = ['FloorPlan{}_physics'.format(i) for i in SCENE_INDICES]


# ENV_ARGS = dict(gridSize=0.25,
#                 width=224, height=224, agentMode='arm', fieldOfView=100,
#                 agentControllerType='mid-level',
#                 server_class=ai2thor.fifo_server.FifoServer,
#                 useMassThreshold = True, massThreshold = 10,
#                 autoSimulation=False, autoSyncTransforms=True,
#                 )

#Functions

def is_object_at_position(controller, action_detail):
    objectId = action_detail['objectId']
    position = action_detail['position']
    current_object_position = get_object_details(controller, objectId)['position']
    return two_dict_equal(dict(position=position), dict(position=current_object_position))

def is_agent_at_position(controller, action_detail):
    # dict(action='TeleportFull', x=initial_location['x'], y=initial_location['y'], z=initial_location['z'], rotation=dict(x=0, y=initial_rotation, z=0), horizon=horizon, standing=True)
    target_pose = dict(
        position={'x': action_detail['x'], 'y': action_detail['y'], 'z': action_detail['z'], },
        rotation=action_detail['rotation'],
        horizon=action_detail['horizon']
    )
    current_agent_pose = controller.last_event.metadata['agent']
    current_agent_pose = dict(
        position=current_agent_pose['position'],
        rotation=current_agent_pose['rotation'],
        horizon=current_agent_pose['cameraHorizon'],
    )
    return two_dict_equal(current_agent_pose, target_pose)


def get_object_details(controller, obj_id):
    return [o for o in controller.last_event.metadata['objects'] if o['objectId'] == obj_id][0]

def initialize_arm(controller, scene_starting_cheating_locations):
    # for start arm from high up as a cheating, this block is very important. never remove
    scene = controller.last_event.metadata['sceneName']
    initial_pose = scene_starting_cheating_locations[scene]
    event1 = controller.step(dict(action='TeleportFull', standing=True, x=initial_pose['x'], y=initial_pose['y'], z=initial_pose['z'], rotation=dict(x=0, y=initial_pose['rotation'], z=0), horizon=initial_pose['horizon']))
    event2 = controller.step(dict(action='MoveArm',  position=dict(x=0.0, y=0, z=0.35), **ADITIONAL_ARM_ARGS))
    event3 = controller.step(dict(action='MoveArmBase', y=0.8, **ADITIONAL_ARM_ARGS))
    return event1, event2, event3

def make_all_objects_unbreakable(controller):
    all_breakable_objects = [o['objectType'] for o in controller.last_event.metadata['objects'] if o['breakable'] is True]
    all_breakable_objects = set(all_breakable_objects)
    for obj_type in all_breakable_objects:
        controller.step(action='MakeObjectsOfTypeUnbreakable', objectType=obj_type)


def reset_the_scene_and_get_reachables(controller, scene_name=None, scene_options=None):
    if scene_name is None:
        if scene_options is None:
            scene_options = SCENE_NAMES
        scene_name = random.choice(scene_options)
    controller.reset(scene_name)
    controller.step(action='MakeAllObjectsMoveable')
    controller.step(action='MakeObjectsStaticKinematicMassThreshold')
    make_all_objects_unbreakable(controller)
    return get_reachable_positions(controller)

def only_reset_scene(controller, scene_name):
    controller.reset(scene_name)
    controller.step(action='MakeAllObjectsMoveable')
    controller.step(action='MakeObjectsStaticKinematicMassThreshold')
    make_all_objects_unbreakable(controller)

def transport_wrapper(controller, target_object, target_location):
    action_detail_list = []
    transport_detail = dict(action = 'PlaceObjectAtPoint', objectId=target_object, position=target_location, forceKinematic=True)
    event = controller.step(**transport_detail)
    action_detail_list.append(transport_detail)
    # controller.step('PhysicsSyncTransforms')
    advance_detail = dict(action='AdvancePhysicsStep', simSeconds=1.0)
    controller.step(**advance_detail)
    action_detail_list.append(advance_detail)
    return event, action_detail_list

def get_parent_receptacles(event, target_obj):
    all_containing_receptacle = set([])
    parent_queue = [target_obj]
    while(len(parent_queue) > 0):
        top_queue = parent_queue[0]
        parent_queue = parent_queue[1:]
        if top_queue in all_containing_receptacle:
            continue
        current_parent_list = event.get_object(top_queue)['parentReceptacles']
        if current_parent_list is None:
            continue
        else:
            parent_queue += current_parent_list
            all_containing_receptacle.update(set(current_parent_list))
    return all_containing_receptacle

def is_object_in_receptacle(event,target_obj,target_receptacle):
    all_containing_receptacle = get_parent_receptacles(event, target_obj)
    return target_receptacle in all_containing_receptacle

def get_reachable_positions(controller):
    something
    ForkedPdb().set_trace() #TODO just to check who is calling this
    event = controller.step('GetReachablePositions')
    # reachable_positions = event.metadata['reachablePositions']
    reachable_positions = event.metadata['actionReturn']

    # if reachable_positions is None or len(reachable_positions) == 0:
    #     reachable_positions = event.metadata['actionReturn']
    if reachable_positions is None or len(reachable_positions) == 0:
        print('Scene name', controller.last_event.metadata['sceneName'])
        pdb.set_trace()
    return reachable_positions
def execute_command(controller, command,action_dict_addition=ADITIONAL_ARM_ARGS):

    base_position = get_current_arm_state(controller)
    change_height = ARM_MOVE_CONSTANT
    change_value = change_height
    action_details = {}

    if command == 'w':
        base_position['z'] += change_value
    elif command == 'z':
        base_position['z'] -= change_value
    elif command == 's':
        base_position['x'] += change_value
    elif command == 'a':
        base_position['x'] -= change_value
    elif command == '3':
        base_position['y'] += change_value
    elif command == '4':
        base_position['y'] -= change_value
    elif command == 'u':
        base_position['h'] += change_height
    elif command == 'j':
        base_position['h'] -= change_height
    elif command == '/':
        action_details = dict('')
        pickupable = controller.last_event.metadata['arm']['pickupableObjects']
        print(pickupable)
    elif command == 'd':
        event = controller.step(action='ReleaseObject')
        action_details = dict(action='ReleaseObject')
    elif command == 'm':
        action_dict_addition = copy.deepcopy(action_dict_addition)
        event = controller.step(action='MoveAgent', ahead=0.2,**action_dict_addition)
        action_details = dict(action='MoveAgent', ahead=0.2,**action_dict_addition)

    elif command == 'r':
        action_dict_addition = copy.deepcopy(action_dict_addition)
        event = controller.step(action='RotateAgent', degrees = 45,**action_dict_addition)
        action_details = dict(action='RotateAgent', degrees = 45,**action_dict_addition)
    elif command == 'l':
        action_dict_addition = copy.deepcopy(action_dict_addition)
        event = controller.step(action='RotateAgent', degrees = -45,**action_dict_addition)
        action_details = dict(action='RotateAgent', degrees = -45,**action_dict_addition)
    elif command == 'p':
        event = controller.step(action='PickupObject')
        action_details = dict(action='PickupObject')
    elif '!' in command and command[0] == '!':
        radius = command.replace('!', '')
        radius = float(radius)
        event = controller.step(action='SetHandSphereRadius', radius=radius)
        action_details = dict(action='SetHandSphereRadius', radius=radius)
    elif command == 'q':
        action_details = {}
    else:
        action_details = {}

    if command in ['w', 'z', 's', 'a', '3', '4']:

        event = controller.step(action='MoveArm', position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
        action_details=dict(action='MoveArm', position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
        success = event.metadata['lastActionSuccess']


    elif command in ['u', 'j']:

        event = controller.step(action='MoveArmBase', y=base_position['h'],**action_dict_addition)
        action_details=dict(action='MoveArmBase', y=base_position['h'],**action_dict_addition)

        success = event.metadata['lastActionSuccess']

    return action_details

def get_current_arm_state(controller):
    h_min = 0.450998873
    h_max = 1.8009994
    agent_base_location = 0.9009995460510254
    event = controller.last_event
    offset = event.metadata['agent']['position']['y'] - agent_base_location
    h_max += offset
    h_min += offset
    joints=(event.metadata['arm']['joints'])
    arm=joints[-1]
#     assert arm['name'] == 'robot_arm_4_jnt'
    xyz_dict = copy.deepcopy(arm['rootRelativePosition'])
    height_arm = joints[0]['position']['y']
    xyz_dict['h'] = (height_arm - h_min) / (h_max - h_min)
    #     print_error([x['position']['y'] for x in joints])
    return xyz_dict

def two_list_equal(l1, l2, threshold, ignore_keys, verbose):
    dict1 = {i: v for (i,v) in enumerate(l1)}
    dict2 = {i: v for (i,v) in enumerate(l2)}
    return two_dict_equal(dict1, dict2, threshold, ignore_keys, verbose)


def get_current_full_state(controller):
    return {'agent_position':controller.last_event.metadata['agent']['position'], 'agent_rotation':controller.last_event.metadata['agent']['rotation'], 'arm_state': controller.last_event.metadata['arm']['joints'], 'held_object': controller.last_event.metadata['arm']['HeldObjects']}


def two_dict_equal(dict1, dict2, threshold=0.001, ignore_keys=[], verbose=False):
    if len(dict1) != len(dict2):
        if verbose:
            print('different len', dict1, dict2)
        return False
    # assert len(dict1) == len(dict2), print('different len', dict1, dict2)
    equal = True
    for k in dict1:
        if k in ignore_keys:
            continue
        val1 = dict1[k]
        val2 = dict2[k]
        if not (type(val1) == type(val2) or (type(val1) in [int, float] and type(val2) in [int, float])):
            if verbose:
                print('different type', dict1, dict2)
            return False
        # assert type(val1) == type(val2) or (type(val1) in [int, float] and type(val2) in [int, float]), ()
        if type(val1) == dict:
            equal = two_dict_equal(val1, val2, threshold = threshold, verbose=verbose, ignore_keys=ignore_keys)
        elif type(val1) == list:
            equal = two_list_equal(val1, val2 , threshold, ignore_keys, verbose)
        # elif val1 != val1: # Either nan or -inf
        #     equal = val2 != val2
        elif type(val1) == float:
            equal = abs(val1 - val2) < threshold
        else:
            equal = (val1 == val2)
        if not equal:
            if verbose:
                print('not equal', 'key', k, 'values', val1, val2)
            return equal
    return equal

def find_arm_distance_to_obj(controller, object_type):
    object_location = controller.last_event.objects_by_type(object_type)[0]['position']
    hand_location = controller.last_event.metadata['arm']['joints'][-1]['position']
    distance = sum([(hand_location[k] - object_location[k]) ** 2 for k in hand_location])**0.5
    return distance

# def close_enough(current_obj_pose, init_obj_pose, threshold):
#     position_close = [
#         abs(current_obj_pose["position"][k] - init_obj_pose["position"][k])
#         <= threshold
#         for k in ["x", "y", "z"]
#     ]
#     position_is_close = sum(position_close) == 3
#     return position_is_close

def get_objects_moved(controller, initial_object_locations, only_visible=False, only_active_moving=False):
    current_object_locations = get_current_object_locations(controller)
    moved_objects = []
    held_objects = controller.last_event.metadata['arm']['heldObjects']

    # moving_objects = [k['objectId'] for k in controller.last_event.metadata['objects'] if k['isMoving'] and k['objectId'] not in held_objects and (k['visible'] or initial_object_locations[k['objectId']]['visible'])]
    # return moving_objects

    for object_id in current_object_locations.keys():
        if object_id not in initial_object_locations:
            print('This is messed up', object_id, 'not exists')
            continue

        if not close_enough(
                current_object_locations[object_id],
                initial_object_locations[object_id],
                threshold=OBJECT_MOVEMENT_THR,
        ):
            if not current_object_locations[object_id]['visible'] and not initial_object_locations[object_id]['visible'] and only_visible:
                continue
            if object_id in held_objects:
                continue
            if only_active_moving and not controller.last_event.get_object(object_id)['isMoving']:
                continue
            moved_objects.append(object_id)

    return moved_objects

def get_current_object_locations(controller):
    obj_loc_dict = {}
    metadata = controller.last_event.metadata["objects"]
    for o in metadata:
        obj_loc_dict[o["objectId"]] = dict(
            position=o["position"], rotation=o["rotation"], visible=o['visible']
        )
    return copy.deepcopy(obj_loc_dict)