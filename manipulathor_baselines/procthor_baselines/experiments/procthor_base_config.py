import platform
import random

from manipulathor_baselines.bring_object_baselines.experiments.ithor.bring_object_ithor_base import \
    BringObjectiThorBaseConfig
from scripts.dataset_generation.find_categories_to_use import FULL_LIST_OF_OBJECTS
from utils.procthor_utils.procthor_helper import PROCTHOR_INVALID_SCENES


class BringObjectProcThorBaseConfig(BringObjectiThorBaseConfig):

    train_scene_indices = [i for i in range(7000)]
    if platform.system() == "Darwin":
        train_scene_indices = [i for i in range(100)]

    valid_rooms = [i for i in train_scene_indices if i not in PROCTHOR_INVALID_SCENES]
    TRAIN_SCENES = [f'ProcTHOR{i}' for i in valid_rooms]


    TEST_SCENES = [f'ProcTHOR{i}' for i in range(1)]
    OBJECT_TYPES = list(set([v for room_typ, obj_list in FULL_LIST_OF_OBJECTS.items() for v in obj_list]))


    random.shuffle(TRAIN_SCENES)