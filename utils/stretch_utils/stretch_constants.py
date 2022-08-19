import ai2thor
import ai2thor.fifo_server

# INTEL_CAMERA_WIDTH = int(720/3)
# INTEL_CAMERA_HEIGHT = int(1280/3)
from manipulathor_utils.debugger_util import ForkedPdb

INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 224,224 # TODO this is too small 320,320
# STRETCH_MANIPULATHOR_COMMIT_ID = '7184aa455bc21cc38406487a3d8e2d65ceba2571'
# STRETCH_MANIPULATHOR_COMMIT_ID = 'f698c1c27a39536858c854cae413fd31987cdf2a'
# STRETCH_MANIPULATHOR_COMMIT_ID = 'fe005524939307669392dab264a22da8ab6ed53a' #This one was rotated
# STRETCH_MANIPULATHOR_COMMIT_ID = '546c50bfa7cfbcec7d5224527e48e6ccb7ed26c2' # just for the segmentation sanity check
# STRETCH_MANIPULATHOR_COMMIT_ID = 'cf23e657aa4738324d09cc79d5f78ea741bf20bf' # eriic commit?--
# STRETCH_MANIPULATHOR_COMMIT_ID  = 'cad761834abd6d0715bbf45c712fbd4947f43710' #new default camera params
# STRETCH_MANIPULATHOR_COMMIT_ID = 'fcd84991b795e7fee29733bb41af09932572baf6' #smaller fov
STRETCH_MANIPULATHOR_COMMIT_ID = '09b6ccf74558395a231927dee8be3b8c83b52ef7' #bigger fov
# PROCTHOR_COMMIT_ID = '0eddca46783a788bfce69b146c496d931c981ae4'
PROCTHOR_COMMIT_ID = '996a369b5484c7037d3737906be81b84a52473a0' #after the arm destroy bug
UPDATED_PROCTHOR_COMMIT_ID = 'fd95db6135273a8ab2e35d4d350dd556c8ced655'
NANNA_COMMIT_ID = '7920893d8e9e44289df95106963e43465928b52d'#'43de4b6a4697b482b88fcdf250f82a06d52169e8'

STRETCH_ENV_ARGS = dict(
    gridSize=0.25,
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
    visibilityDistance=1.0,
    # fieldOfView=42,
    # fieldOfView=69,
    fieldOfView=69,
    agentControllerType="mid-level",
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=True,
    agentMode='stretch',
    renderDepthImage=True,
)
# if depth to rgb:
KINECT_REAL_W, KINECT_REAL_H = 720, 1280
KINECT_RESIZED_W, KINECT_RESIZED_H = 180, 320
KINECT_FOV_W, KINECT_FOV_H = 59, 90

# if rgb to depth:
# KINECT_REAL_W, KINECT_REAL_H = 576, 640
# KINECT_RESIZED_W, KINECT_RESIZED_H = 288, 320
# KINECT_FOV_W, KINECT_FOV_H = 65, 75

INTEL_REAL_W, INTEL_REAL_H = 1920, 1080
INTEL_RESIZED_W, INTEL_RESIZED_H = 320, 180
INTEL_FOV_W, INTEL_FOV_H = 69, 42

# MIN_INTEL_DEPTH = 0.28
MIN_INTEL_DEPTH = 0
MAX_INTEL_DEPTH = 3
# MIN_KINECT_DEPTH = 0.5
MIN_KINECT_DEPTH = 0.
MAX_KINECT_DEPTH = 3.86

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
ROTATE_RIGHT_SMALL = 'RotateRightSmall'
ROTATE_LEFT_SMALL = 'RotateLeftSmall'
MOVE_WRIST_P_SMALL = 'MoveWristPSmall'
MOVE_WRIST_M_SMALL = 'MoveWristMSmall'

ADITIONAL_ARM_ARGS = {
    "disableRendering": True,
    "returnToStart": True,
    "speed": 1,
}

MOVE_ARM_CONSTANT = 0.05
ARM_LENGTH = 1.

# and as far as your earlier question regarding (H fov vs V fov)
# the field of view that gets set through the API corresponds to the vertical field of view (https://docs.unity3d.com/ScriptReference/Camera-fieldOfView.html)
