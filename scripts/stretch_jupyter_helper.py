import copy
import math
import random
import ai2thor
import pdb
import ai2thor.fifo_server


### CONSTANTS
from pyquaternion import Quaternion

from manipulathor_utils.debugger_util import ForkedPdb

ADITIONAL_ARM_ARGS = {
    'disableRendering': True,
    'returnToStart': True,
    'speed': 1,
}

ARM_MOVE_CONSTANT = 0.05
WRIST_ROTATION = 10 #TODO we might have to make this smalelr tbh

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

    only_reset_scene(controller, scene_name)
    return get_reachable_positions(controller)

def reset_environment_and_additional_commands(controller, scene_name):
    controller.reset(scene_name)
    controller.step(action="MakeAllObjectsMoveable")
    controller.step(action="MakeObjectsStaticKinematicMassThreshold")
    make_all_objects_unbreakable(controller)

    event_init_arm = controller.step(dict(action="MoveArmBase", y=0.8, **ADITIONAL_ARM_ARGS))
    if event_init_arm.metadata['lastActionSuccess'] is False:
        print('Initialze arm failed')
    return

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
    event = controller.step('GetReachablePositions')
    # reachable_positions = event.metadata['reachablePositions']
    reachable_positions = event.metadata['actionReturn']

    # if reachable_positions is None or len(reachable_positions) == 0:
    #     reachable_positions = event.metadata['actionReturn']
    if reachable_positions is None or len(reachable_positions) == 0:
        print('ERRRRROOOOOORR: Scene name', controller.last_event.metadata['sceneName'])
        # pdb.set_trace()
    return reachable_positions
def execute_command(controller, command,action_dict_addition):

    base_position = get_current_arm_state(controller)
    # base_position = dict(x=0, y=0, z=0)
    change_height = ARM_MOVE_CONSTANT
    change_value = change_height
    action_details = {}

    if command == 'hu':
        base_position['y'] += change_value
    elif command == 'hd':
        base_position['y'] -= change_value
    elif command == 'ao':
        base_position['z'] += change_value
    elif command == 'ai':
        base_position['z'] -= change_value
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

    elif command == 'b':
        action_dict_addition = copy.deepcopy(action_dict_addition)
        event = controller.step(action='MoveAgent', ahead=-0.2,**action_dict_addition)
        action_details = dict(action='MoveAgent', ahead=-0.2,**action_dict_addition)

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
    elif command == 'wp':
        event = controller.step(action='RotateWristRelative', yaw=-WRIST_ROTATION)
        action_details = dict(action='RotateWristRelative', yaw=-WRIST_ROTATION)
    elif command == 'wn':
        event = controller.step(action='RotateWristRelative', yaw=WRIST_ROTATION)
        action_details = dict(action='RotateWristRelative', yaw=WRIST_ROTATION)
    else:
        action_details = {}

    if command in ['hu', 'hd', 'ao', 'ai']:

        event = controller.step(action='MoveArm', position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
        action_details=dict(action='MoveArm', position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
        # TODO this does not work
        # event = controller.step(action='MoveArm', coordinateSpace="wrist", position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
        # action_details=dict(action='MoveArm', coordinateSpace="wrist", position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
        success = event.metadata['lastActionSuccess']



    return action_details

def get_current_wrist_state(controller):
    arm = controller.last_event.metadata['arm']['joints'][-1]
    rotations = arm['rootRelativeRotation']
    quaternion = Quaternion(axis=[rotations['x'], rotations['y'], rotations['z']], degrees=rotations['w'])
    return quaternion


def get_current_arm_state(controller):

    arm = controller.last_event.metadata['arm']['joints'] #TODO is this the right one? how about wrist movements
    z = arm[-1]['rootRelativePosition']['z']
    x = 0 #arm[-1]['rootRelativePosition']['x']
    y = arm[0]['rootRelativePosition']['y'] - 0.16297650337219238 #TODO?
    return dict(x=0,y=y, z=z)

def two_list_equal(l1, l2):
    dict1 = {i: v for (i,v) in enumerate(l1)}
    dict2 = {i: v for (i,v) in enumerate(l2)}
    return two_dict_equal(dict1, dict2)


def get_current_full_state(controller):
    return {'agent_position':controller.last_event.metadata['agent']['position'], 'agent_rotation':controller.last_event.metadata['agent']['rotation'], 'arm_state': controller.last_event.metadata['arm']['joints'], 'held_object': controller.last_event.metadata['arm']['HeldObjects']}


def two_dict_equal(dict1, dict2, threshold=0.001, ignore_keys=[]):
    if len(dict1) != len(dict2):
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
            print('different type', dict1, dict2)
            return False
        # assert type(val1) == type(val2) or (type(val1) in [int, float] and type(val2) in [int, float]), ()
        if type(val1) == dict:
            equal = two_dict_equal(val1, val2)
        elif type(val1) == list:
            equal = two_list_equal(val1, val2)
        # elif val1 != val1: # Either nan or -inf
        #     equal = val2 != val2
        elif type(val1) == float:
            equal = abs(val1 - val2) < threshold
        else:
            equal = (val1 == val2)
        if not equal:
            print('not equal', 'key', k, 'values', val1, val2)
            return equal
    return equal

def find_arm_distance_to_obj(controller, object_type):
    object_location = controller.last_event.objects_by_type(object_type)[0]['position']
    hand_location = controller.last_event.metadata['arm']['joints'][-1]['position']
    distance = sum([(hand_location[k] - object_location[k]) ** 2 for k in hand_location])**0.5
    return distance

# def old_execute_command(controller, command,action_dict_addition):
#
#     base_position = get_current_arm_state(controller)
#     change_height = ARM_MOVE_CONSTANT
#     change_value = change_height
#     action_details = {}
#
#     if command == 'hu':
#         base_position['y'] += change_value
#     elif command == 'hd':
#         base_position['y'] -= change_value
#     elif command == 'ao':
#         base_position['z'] += change_value
#     elif command == 'ai':
#         base_position['z'] -= change_value
#     elif command == '/':
#         action_details = dict('')
#         pickupable = controller.last_event.metadata['arm']['pickupableObjects']
#         print(pickupable)
#     elif command == 'd':
#         event = controller.step(action='ReleaseObject')
#         action_details = dict(action='ReleaseObject')
#     elif command == 'm':
#         action_dict_addition = copy.deepcopy(action_dict_addition)
#         event = controller.step(action='MoveAgent', ahead=0.2,**action_dict_addition)
#         action_details = dict(action='MoveAgent', ahead=0.2,**action_dict_addition)
#
#     elif command == 'b':
#         action_dict_addition = copy.deepcopy(action_dict_addition)
#         event = controller.step(action='MoveAgent', ahead=-0.2,**action_dict_addition)
#         action_details = dict(action='MoveAgent', ahead=-0.2,**action_dict_addition)
#
#     elif command == 'r':
#         action_dict_addition = copy.deepcopy(action_dict_addition)
#         event = controller.step(action='RotateAgent', degrees = 45,**action_dict_addition)
#         action_details = dict(action='RotateAgent', degrees = 45,**action_dict_addition)
#     elif command == 'l':
#         action_dict_addition = copy.deepcopy(action_dict_addition)
#         event = controller.step(action='RotateAgent', degrees = -45,**action_dict_addition)
#         action_details = dict(action='RotateAgent', degrees = -45,**action_dict_addition)
#     elif command == 'p':
#         event = controller.step(action='PickupObject')
#         action_details = dict(action='PickupObject')
#     elif '!' in command and command[0] == '!':
#         radius = command.replace('!', '')
#         radius = float(radius)
#         event = controller.step(action='SetHandSphereRadius', radius=radius)
#         action_details = dict(action='SetHandSphereRadius', radius=radius)
#     elif command == 'q':
#         action_details = {}
#     else:
#         action_details = {}
#
#     if command in ['hu', 'hd', 'ao', 'ai']:
#
#         event = controller.step(action='MoveArm', position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
#         action_details=dict(action='MoveArm', position=dict(x=base_position['x'], y=base_position['y'], z=base_position['z']),**action_dict_addition)
#         success = event.metadata['lastActionSuccess']
#
#
#
#     return action_details
