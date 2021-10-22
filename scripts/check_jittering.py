import copy
import pdb

import ai2thor.controller

from ithor_arm.ithor_arm_constants import ENV_ARGS
from scripts.jupyter_helper import two_dict_equal

# commit_id = '58bf22c0b9aa0d3abe5fd8c3b43479ecc8d2a228'
commit_id = '2f8dd9f95e4016db60155a0cc18b834a6339c8e1' #TODO change everywhere
ENV_ARGS['commit_id'] = commit_id
controller = ai2thor.controller.Controller(**ENV_ARGS)

kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

scenes = kitchens + living_rooms + bedrooms + bathrooms

def make_dict_from_object(object_list):
    result = {}
    for obj in object_list:
        # result[obj['objectId']] = dict(position=obj['position'], rotation=obj['rotation'])
        result[obj['objectId']] = dict(position=obj['position'])
    return result

for scene in scenes:
    # print(scene)
    controller.reset(scene)
    last_moved = -1
    total = 400
    for i in range(total):
        if last_moved < 0:
            initial_object = make_dict_from_object(controller.last_event.metadata['objects'])
        controller.step('AdvancePhysicsStep')
        # controller.step('AdvancePhysicsStep', simSeconds=200)
        if False:
            last_moved = 1
        else:
            final_object = make_dict_from_object(controller.last_event.metadata['objects'])
            equal = two_dict_equal(final_object, initial_object, threshold=0.02, ignore_keys=[], verbose=False)
            if equal:
                last_moved += 1
            else:
                last_moved = -1

    final_object = make_dict_from_object(controller.last_event.metadata['objects'])
    equal = two_dict_equal(final_object, initial_object, threshold=0.02, ignore_keys=[], verbose=False)
    # if not equal:
    #     print(scene)
    #     for obj in initial_object:
    #         if not two_dict_equal(initial_object[obj], final_object[obj], threshold=0.02, verbose=False):
    #             print(obj)
    # continue TODO remove
    equal = two_dict_equal(final_object, initial_object, threshold=0.02, ignore_keys=[], verbose=True)
    print(scene)
    print(total - last_moved)
    if last_moved < 200:
        print('Oh NOOOO', last_moved)

