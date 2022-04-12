import copy
import pdb
import random

from ai2thor.controller import Controller
import datasets
import pickle

from scripts.test_stretch import visualize
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS

env_to_work_with = copy.deepcopy(STRETCH_ENV_ARGS)
# STRETCH_ENV_ARGS = dict(
#     gridSize=0.25,
#     width=INTEL_CAMERA_WIDTH,
#     height=INTEL_CAMERA_HEIGHT,
#     visibilityDistance=1.0,
#     # fieldOfView=42,
#     # fieldOfView=69,
#     fieldOfView=69,
#     agentControllerType="mid-level",
#     server_class=ai2thor.fifo_server.FifoServer,
#     useMassThreshold=True,
#     massThreshold=10,
#     autoSimulation=False,
#     autoSyncTransforms=True,
#     renderInstanceSegmentation=True,
#     agentMode='stretch',
#     renderDepthImage=True,
# )
env_to_work_with['branch'] = 'nanna'
env_to_work_with['scene'] = 'Procedural'

controller = Controller(**env_to_work_with)
house_dataset = datasets.load_dataset("allenai/houses", use_auth_token=True)

# Load the first house
house_entry = house_dataset["train"][10]
controller.step(action="CreateHouse", house=pickle.loads(house_entry["house"]))
house = pickle.loads(house_entry["house"])
controller.step(action="TeleportFull", **house["metadata"]["agent"])
rp_event = controller.step(action="GetReachablePositions")

reachable_positions = rp_event.metadata["actionReturn"]
random_position = random.choice(reachable_positions)
#TODO set with stretch
#TODO visualize
#TODO how to be efficient
pdb.set_trace()
visualize(controller)
