"""Constant values and hyperparameters that are used by the environment."""
import ai2thor
import ai2thor.fifo_server
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

# MANIPULATHOR_COMMIT_ID = "68212159d78aab5c611b7f16338380993884a06a"
# MANIPULATHOR_COMMIT_ID = 'bcc2e62970823667acb5c2a56e809419f1521e52'

# # Most updated thor version, thor 3.3.1
# MANIPULATHOR_COMMIT_ID = "d26bb0ef75d95074c39718cf9f1a0890ac2c974f"
# Most updated thor version, thor 3.3.4
MANIPULATHOR_COMMIT_ID = "39c4a83cecb2daa36b9786b7017a22dc3485a9ea"

#TODO remove just for exp room luca toggle grasper visibility and potentially a boost in FPS
MANIPULATHOR_COMMIT_ID = "58bf22c0b9aa0d3abe5fd8c3b43479ecc8d2a228"

MOVE_THR = 0.01
ARM_MIN_HEIGHT = 0.450998873
ARM_MAX_HEIGHT = 1.8009994
MOVE_ARM_CONSTANT = 0.05
MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT

ADITIONAL_ARM_ARGS = {
    "disableRendering": True,
    "returnToStart": True,
    "speed": 1,
}

MOVE_AHEAD = "MoveAheadContinuous"
ROTATE_LEFT = "RotateLeftContinuous"
ROTATE_RIGHT = "RotateRightContinuous"
MOVE_ARM_HEIGHT_P = "MoveArmHeightP"
MOVE_ARM_HEIGHT_M = "MoveArmHeightM"
MOVE_ARM_X_P = "MoveArmXP"
MOVE_ARM_X_M = "MoveArmXM"
MOVE_ARM_Y_P = "MoveArmYP"
MOVE_ARM_Y_M = "MoveArmYM"
MOVE_ARM_Z_P = "MoveArmZP"
MOVE_ARM_Z_M = "MoveArmZM"
PICKUP = "PickUpMidLevel"
DONE = "DoneMidLevel"


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
)

TRAIN_OBJECTS = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug"]
TEST_OBJECTS = ["Potato", "Pan", "Egg", "Spatula", "Cup"] # we have to remove soap bottle from categories, "SoapBottle"


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
    # controller.step('ToggleMagnetVisibility') #TODO do we want to have this here or do we want to have it during training and only change it for obejct detection part?
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

with open("datasets/apnd-dataset/starting_pose.json") as f:
    ARM_START_POSITIONS = json.load(f)
