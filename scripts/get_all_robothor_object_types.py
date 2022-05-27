import math
import pdb
import platform

import ai2thor
import copy
import time
import random
import ai2thor.controller

from scripts.dataset_generation.find_categories_to_use import ROBOTHOR_TRAIN
from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS, STRETCH_MANIPULATHOR_COMMIT_ID#, ROBOTHOR_SCENE_NAMES
controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS, commit_id=STRETCH_MANIPULATHOR_COMMIT_ID)
all_object_types = []
for scene in ROBOTHOR_TRAIN:
    controller.reset(scene)
    object_types = [o['objectType'] for o in controller.last_event.metadata['objects']]
    object_types = ((object_types))
    all_object_types += (object_types)
all_object_types = (set(all_object_types))
print(all_object_types)

pdb.set_trace()
