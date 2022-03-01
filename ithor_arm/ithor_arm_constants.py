"""Constant values and hyperparameters that are used by the environment."""
import ai2thor
import ai2thor.fifo_server
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

# MANIPULATHOR_COMMIT_ID = "68212159d78aab5c611b7f16338380993884a06a"
# MANIPULATHOR_COMMIT_ID = 'bcc2e62970823667acb5c2a56e809419f1521e52'

# # Most updated thor version, thor 3.3.1
# MANIPULATHOR_COMMIT_ID = "d26bb0ef75d95074c39718cf9f1a0890ac2c974f"
# Most updated thor version, thor 3.3.4
# MANIPULATHOR_COMMIT_ID = "39c4a83cecb2daa36b9786b7017a22dc3485a9ea"

# for exp room luca toggle grasper visibility and potentially a boost in FPS
# MANIPULATHOR_COMMIT_ID = "58bf22c0b9aa0d3abe5fd8c3b43479ecc8d2a228"
# MANIPULATHOR_COMMIT_ID = '214bc8036f323f4d8418e0a76c4251c401793bd5'
# MANIPULATHOR_COMMIT_ID = '2f8dd9f95e4016db60155a0cc18b834a6339c8e1' #

MANIPULATHOR_COMMIT_ID = '2b5bcec105e30de464f34c4349a40f015b872517' #TODO just for the remove arm visibility

MOVE_THR = 0.01
ARM_MIN_HEIGHT = 0.450998873
ARM_MAX_HEIGHT = 1.8009994
MOVE_ARM_CONSTANT = 0.05
MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT
ARM_LENGTH = 1.
# According to the following it is 0.94 to be precise. But close enough
# controller.reset('FloorPlan2');controller.step('RotateRight');controller.step('MoveArmBase', y=1);controller.step('MoveArm', position=dict(x=0,y=0,z=1));controller.step('Pass');controller.last_event.metadata['arm']['joints'][-1]

ADITIONAL_ARM_ARGS = {
    "disableRendering": True,
    "returnToStart": True,
    "speed": 1,
}

MOVE_AHEAD = "MoveAhead"
MOVE_BACK = "MoveBack"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
MOVE_ARM_HEIGHT_P = "MoveArmHeightP"
MOVE_ARM_HEIGHT_M = "MoveArmHeightM"
MOVE_ARM_X_P = "MoveArmXP"
MOVE_ARM_X_M = "MoveArmXM"
MOVE_ARM_Y_P = "MoveArmYP"
MOVE_ARM_Y_M = "MoveArmYM"
MOVE_ARM_Z_P = "MoveArmZP"
MOVE_ARM_Z_M = "MoveArmZM"
PICKUP = "PickUp"
DONE = "Done"
MOVE_WRIST_P = 'MoveWristP'
MOVE_WRIST_M = 'MoveWristM'
GRASP_O = 'GraspOpen'
GRASP_C = 'GraspClose'

SET_OF_ALL_AGENT_ACTIONS = [MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT, MOVE_ARM_HEIGHT_P, MOVE_ARM_HEIGHT_M, MOVE_ARM_X_P, MOVE_ARM_X_M, MOVE_ARM_Y_P, MOVE_ARM_Y_M, MOVE_ARM_Z_P, MOVE_ARM_Z_M, PICKUP, DONE, MOVE_WRIST_P, MOVE_WRIST_M, GRASP_O, GRASP_C, ]

ENV_ARGS = dict(
    gridSize=0.25,
    width=224,
    height=224,
    visibilityDistance=1.0,
    agentMode="arm",
    fieldOfView=100,
    agentControllerType="mid-level",
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=True,
    commit_id=MANIPULATHOR_COMMIT_ID,
)

ARM_ACTIONS_ORDERED = [MOVE_ARM_HEIGHT_P,MOVE_ARM_HEIGHT_M,MOVE_ARM_X_P,MOVE_ARM_X_M,MOVE_ARM_Y_P,MOVE_ARM_Y_M,MOVE_ARM_Z_P,MOVE_ARM_Z_M,MOVE_AHEAD,ROTATE_RIGHT,ROTATE_LEFT]
ARM_SHORTENED_ACTIONS_ORDERED = ['u','j','s','a','3','4','w','z','m','r','l']

TRAIN_OBJECTS = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug"]
TEST_OBJECTS = ["Potato", "Pan", "Egg", "Spatula", "Cup", 'SoapBottle'] # we have to remove soap bottle from categories, "SoapBottle"



DONT_USE_ALL_POSSIBLE_OBJECTS_EVER = ['ButterKnife', 'Sink', 'GarbageCan', 'Plate', 'PepperShaker', 'Cup', 'Mug', 'CoffeeMachine', 'Knife', 'Spatula', 'Pan', 'Egg', 'Pot', 'Toaster', 'DishSponge', 'Potato', 'Spoon', 'Apple', 'Bread', 'Fork', 'Faucet', 'StoveBurner', 'Lettuce', 'SoapBottle', 'Bowl', 'SaltShaker', 'Tomato']

def make_all_objects_unbreakable(controller):
    all_breakable_objects = [
        o["objectType"]
        for o in controller.last_event.metadata["objects"]
        if o["breakable"] is True
    ]
    all_breakable_objects = set(all_breakable_objects)
    for obj_type in all_breakable_objects:
        controller.step(action="MakeObjectsOfTypeUnbreakable", objectType=obj_type)


def reset_environment_and_additional_commands(controller, scene_name):
    controller.reset(scene_name)
    controller.step(action="MakeAllObjectsMoveable")
    controller.step(action="MakeObjectsStaticKinematicMassThreshold")
    make_all_objects_unbreakable(controller)
    # controller.step('ToggleMagnetVisibility')  do we want to have this here or do we want to have it during training and only change it for obejct detection part?
    # controller.step(action='SetHandSphereRadius', radius=0.2)
    return


def transport_wrapper(controller, target_object, target_location):
    transport_detail = dict(
        action="PlaceObjectAtPoint",
        objectId=target_object,
        position=target_location,
        forceKinematic=True,
    )
    advance_detail = dict(action="AdvancePhysicsStep", simSeconds=1.0)

    if issubclass(type(controller), IThorEnvironment):
        event = controller.step(transport_detail)
        controller.step(advance_detail)
    elif type(controller) == ai2thor.controller.Controller:
        event = controller.step(**transport_detail)
        controller.step(**advance_detail)
    return event


VALID_OBJECT_LIST = [
    "Knife",
    "Bread",
    "Fork",
    "Potato",
    "SoapBottle",
    "Pan",
    "Plate",
    "Tomato",
    "Egg",
    "Pot",
    "Spatula",
    "Cup",
    "Bowl",
    "SaltShaker",
    "PepperShaker",
    "Lettuce",
    "ButterKnife",
    "Apple",
    "DishSponge",
    "Spoon",
    "Mug",
]

import json
try:
    with open("datasets/apnd-dataset/starting_pose.json") as f:
        ARM_START_POSITIONS = json.load(f)
except Exception:
    print('Couldnt find initial poses')
