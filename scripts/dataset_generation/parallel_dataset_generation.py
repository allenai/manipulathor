import argparse
import os
import pdb
import sys
import threading
import time


# index = 5
# # for index 6 to 8 run later
# list_of_scenes = ['FloorPlan{}_physics'.format(str(i)) for i in range(index * 5 + 2, (index + 1) * 5 + 2)]
# list_of_objects = ['Apple', 'Bread', 'Book', 'Tomato', 'Lettuce', 'Pot', 'Mug']
from scripts.dataset_generation.find_categories_to_use import SCENE_TYPE_TO_IDS, FULL_LIST_OF_OBJECTS

NUM_THREADS = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Data loader')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--restart', default=-1, type=int)
    args = parser.parse_args()
    assert args.restart == -1 or (args.restart - 1) % NUM_THREADS == args.seed
    if args.seed not in [i for i in range(NUM_THREADS)]:
        raise Exception('Does not work', args.seed)
    return args

args=parse_args()
# list_of_scenes = ['FloorPlan{}_physics'.format(str(i)) for i in range(1, 31) if ((i - 1) % NUM_THREADS == args.seed)]
# if args.restart > 0:
#     list_of_scenes = ['FloorPlan{}_physics'.format(str(i)) for i in range(args.restart, 31) if ((i - 1) % NUM_THREADS == args.seed)]



def start_exp(scene, obj):
    print('scene', scene, 'obj', obj)
    command = 'python3 scripts/dataset_generation/make_dataset.py --verbose --visualize --object_type {} --scene_name {}'.format(obj, scene)
    print(command)
    os.system(command)
    print('Finished', scene, obj)

def get_scene_number(scene):
    return int(scene.replace('FloorPlan', '').replace('_physics', ''))

for scene in ['living_rooms', 'bedrooms', 'bathrooms']:
    list_of_scenes = SCENE_TYPE_TO_IDS[scene]
    list_of_objects = FULL_LIST_OF_OBJECTS[scene]
    for scene in list_of_scenes:
        for obj in (list_of_objects):

            scene_number = get_scene_number(scene)
            if scene_number % NUM_THREADS != args.seed:
                continue
            start_exp(scene, obj)