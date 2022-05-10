import copy
import json
import os.path
import pdb
import random

file = 'datasets/apnd-dataset/stretch_init_location.json'
with open(file) as f:
    init_locations = json.load(f)

dest_file = 'datasets/apnd-dataset/deterministic_stretch_init_location.json'
deterministic_locations = {}

for k in init_locations:
    list_of_poses = copy.deepcopy(init_locations[k])
    random.shuffle(list_of_poses)
    deterministic_locations[k] = list_of_poses
if os.path.exists(dest_file):
    raise Exception('It exists already')
with open(dest_file, 'w') as f:
    json.dump(deterministic_locations, f)