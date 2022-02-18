import json
import random

import ai2thor
import ai2thor.controller

from scripts.dataset_generation.find_categories_to_use import ROBOTHOR_SCENE_NAMES
from scripts.stretch_jupyter_helper import get_reachable_positions
from utils.stretch_utils.stretch_constants import STRETCH_MANIPULATHOR_COMMIT_ID, STRETCH_ENV_ARGS

STRETCH_ENV_ARGS['commit_id'] = STRETCH_MANIPULATHOR_COMMIT_ID
STRETCH_ENV_ARGS['renderDepthImage'] = False
STRETCH_ENV_ARGS['renderInstanceSegmentation'] = False
controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)

kitchens = [f"FloorPlan{i}_physics" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}_physics" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}_physics" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}_physics" for i in range(1, 31)]

scenes = kitchens + living_rooms + bedrooms + bathrooms + ROBOTHOR_SCENE_NAMES


dict_to_write = {}
for room in scenes:
    print(room)
    controller.reset(room)
    print(controller._build.url)
    reachable_positions = get_reachable_positions(controller)
    dict_to_write.setdefault(room, [])
    for initial_pose in reachable_positions:
        initial_pose['rotation'] = random.choice([i for i in range(0,360,30)])
        initial_pose['horizon'] = 20
        event1 = controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=initial_pose["x"],
                y=initial_pose["y"],
                z=initial_pose["z"],
                rotation=dict(x=0, y=initial_pose["rotation"], z=0),
                horizon=initial_pose["horizon"],
            )
        )
        if event1.metadata['lastActionSuccess'] == True:
            something_succeeded = False
            for action in [{'action':'MoveAgent', 'ahead':0.1},{'action':'MoveAgent', 'ahead':0.1},{'action':'RotateAgent', 'degrees':10},{'action':'RotateAgent', 'degrees':-10},]:
                event = controller.step(**action)
                if event.metadata['lastActionSuccess'] == True:
                    something_succeeded = True
                    break
            if something_succeeded:
                dict_to_write[room].append(initial_pose)
    random.shuffle(dict_to_write[room])
    print('for scene', room, 'total', len(dict_to_write[room]), 'out of', len(reachable_positions))

with open('datasets/apnd-dataset/stretch_init_location.json', 'w') as f:
    json.dump(dict_to_write, f)

