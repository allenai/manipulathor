import copy
import datetime
import pdb
import random

from ai2thor.controller import Controller
import datasets
import pickle

from scripts.stretch_jupyter_helper import make_all_objects_unbreakable, execute_command
from scripts.test_stretch import visualize, manual_task
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, PROCTHOR_COMMIT_ID, ADITIONAL_ARM_ARGS

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
# env_to_work_with['branch'] = 'nanna' #
env_to_work_with['scene'] = 'Procedural'
env_to_work_with['commit_id'] = PROCTHOR_COMMIT_ID

controller = Controller(**env_to_work_with)
house_dataset = datasets.load_dataset("allenai/houses", use_auth_token=True)

all_house_ind = [i for i in range(len(house_dataset["train"]))]
random.seed(datetime.datetime.now().microsecond)
random.shuffle(all_house_ind)
for house_ind in all_house_ind:
    # Load the  house
    print('In house_ind',house_ind)
    house_entry = house_dataset["train"][house_ind]
    controller.reset()
    controller.step(action="CreateHouse", house=pickle.loads(house_entry["house"]))

    house = pickle.loads(house_entry["house"])
    controller.step(action="TeleportFull", **house["metadata"]["agent"])
    rp_event = controller.step(action="GetReachablePositions")

    controller.step(action="MakeAllObjectsMoveable")
    controller.step(action="MakeObjectsStaticKinematicMassThreshold")
    make_all_objects_unbreakable(controller)
    event_init_arm = controller.step(dict(action="MoveArm", position=dict(x=0,y=0.8,z=0), **ADITIONAL_ARM_ARGS))
    if event_init_arm.metadata['lastActionSuccess'] is False:
        print(f"Bring arm up in initialization failed in {house_ind}")
        continue

    reachable_positions = rp_event.metadata["actionReturn"]
    if reachable_positions is None or len(reachable_positions) == 0:
        print('reachable position bnot found', reachable_positions)
        continue
    random_position = random.choice(reachable_positions)
    all_actions = []
    before = datetime.datetime.now()
    while(len(all_actions) < 200):
        ALL_POSSIBLE_STRETCH_ACTIONS = ['m','r','l','p','wp','wm','zp','zm','hp','hm', 'd', 'b']
        action = random.choice(ALL_POSSIBLE_STRETCH_ACTIONS)
        if len(controller.last_event.metadata['arm']['pickupableObjects']) > 0:
            action = 'p'
        detail = execute_command(controller, action, ADITIONAL_ARM_ARGS)
        all_actions.append(detail)
        if action == 'p' and len(controller.last_event.metadata['arm']['heldObjects']) > 0:
            print('We are holding an object ', controller.last_event.metadata['arm']['heldObjects'])
        if not controller.last_event.metadata['lastActionSuccess']:
            random.shuffle(ALL_POSSIBLE_STRETCH_ACTIONS)
            is_stuck = True
            for action in ALL_POSSIBLE_STRETCH_ACTIONS:
                detail = execute_command(controller, action, ADITIONAL_ARM_ARGS)
                all_actions.append(detail)
                if controller.last_event.metadata['lastActionSuccess']:
                    is_stuck = False
                    break
            if is_stuck:
                print('Agent is stuck')
                print('in house', house_ind)
                print(all_actions)

                break
    after = datetime.datetime.now()
    diff = after - before
    print('FPS is ', len(all_actions) / diff.seconds)
        # manual_task(controller, final=True, init_sequence=[action, 'q'])
        # visualize(controller, save=True)
