import ai2thor
import ai2thor.fifo_server

# INTEL_CAMERA_WIDTH = int(720/3)
# INTEL_CAMERA_HEIGHT = int(1280/3)
INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 224,224 #TODO just for the beginning you have to chngfe this and have different values for differnt cameras
# STRETCH_MANIPULATHOR_COMMIT_ID = '7184aa455bc21cc38406487a3d8e2d65ceba2571'
STRETCH_MANIPULATHOR_COMMIT_ID = 'fe005524939307669392dab264a22da8ab6ed53a' #TODO are we sure about this?
# STRETCH_MANIPULATHOR_COMMIT_ID = 'f698c1c27a39536858c854cae413fd31987cdf2a' #TODO we need a new dataset for this
STRETCH_ENV_ARGS = dict(
    gridSize=0.25,
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
    visibilityDistance=1.0,
    # fieldOfView=42,
    # fieldOfView=69,
    fieldOfView=100, #TODO definitely change
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



# and as far as your earlier question regarding (H fov vs V fov)
# the field of view that gets set through the API corresponds to the vertical field of view (https://docs.unity3d.com/ScriptReference/Camera-fieldOfView.html)
# TODO then 42 would be the correct value probably. Also make sure you set these correctly (and differently for the azure camera)