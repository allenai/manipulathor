import ast
import glob
import json

from utils.stretch_utils.stretch_constants import STRETCH_ENV_ARGS
import math
import pdb
import platform

import ai2thor
import copy
import time
import random
import ai2thor.controller
import datetime
import cv2
import os
import matplotlib.pyplot as plt
import os
import numpy as np
# from utils.mid_level_constants import  scene_start_cheating_init_pose
from pyquaternion import Quaternion

from scripts.jupyter_helper import get_reachable_positions
from scripts.stretch_jupyter_helper import two_dict_equal, ARM_MOVE_CONSTANT, get_current_arm_state, only_reset_scene, transport_wrapper, ADITIONAL_ARM_ARGS, execute_command, WRIST_ROTATION, get_current_wrist_state, reset_environment_and_additional_commands

screen_size=224

STRETCH_ENV_ARGS['width'] = screen_size
STRETCH_ENV_ARGS['height'] = screen_size
STRETCH_ENV_ARGS['agentMode']='stretch'
STRETCH_ENV_ARGS['commit_id']='7184aa455bc21cc38406487a3d8e2d65ceba2571'
# STRETCH_ENV_ARGS['renderDepthImage'] = True

controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)

kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

all_scenes = kitchens + living_rooms + bedrooms + bathrooms
SCENE_TYPE_TO_IDS = {
    'kitchens':kitchens,
    'living_rooms':living_rooms,
    'bedrooms':bedrooms,
    'bathrooms':bathrooms,
}

def generate_files():
    for room_type in SCENE_TYPE_TO_IDS.keys():
        rooms = SCENE_TYPE_TO_IDS[room_type]
        object_types = {}
        for r in rooms:
            controller.reset(r)
            movable_objects = [o['objectType'] for o in controller.last_event.metadata['objects'] if o['pickupable']]
            for obj in movable_objects:
                if movable_objects.count(obj) > 1:
                    print('(', obj, end=') ')
                    continue
                object_types.setdefault(obj, 0)
                object_types[obj] += 1
        print(' ')
        print(room_type, '=',object_types)

kitchens = {'Book': 2, 'Bottle': 7, 'Knife': 30, 'Bread': 30, 'Fork': 30, 'Potato': 30, 'SoapBottle': 30, 'Kettle': 15, 'Pan': 30, 'Plate': 30, 'Tomato': 30, 'Egg': 30, 'CreditCard': 3, 'WineBottle': 10, 'Pot': 30, 'Spatula': 30, 'PaperTowelRoll': 10, 'Cup': 30, 'Bowl': 30, 'SaltShaker': 30, 'PepperShaker': 30, 'Lettuce': 30, 'ButterKnife': 30, 'Apple': 30, 'DishSponge': 30, 'Spoon': 30, 'Mug': 30, 'Statue': 3, 'Ladle': 9, 'CellPhone': 4, 'Pen': 2, 'SprayBottle': 2, 'Pencil': 2}
living_rooms = {'Book': 7, 'Box': 30, 'Statue': 16, 'Laptop': 30, 'TissueBox': 9, 'CreditCard': 30, 'Plate': 10, 'KeyChain': 30, 'Vase': 7, 'Pencil': 4, 'Pillow': 30, 'Bowl': 5, 'RemoteControl': 30, 'Watch': 16, 'Pen': 5, 'Newspaper': 18, 'WateringCan': 13, 'Boots': 5, 'CellPhone': 7, 'Candle': 2}
bedrooms = {'Book': 30, 'Box': 12, 'Laptop': 30, 'CellPhone': 30, 'BaseballBat': 12, 'BasketBall': 11, 'TissueBox': 4, 'CreditCard': 30, 'AlarmClock': 30, 'Pencil': 30, 'Boots': 3, 'Pillow': 18, 'KeyChain': 30, 'Bowl': 13, 'Watch': 2, 'CD': 28, 'Pen': 30, 'Mug': 16, 'Statue': 4, 'TennisRacket': 10, 'TeddyBear': 10, 'Vase': 1, 'Cloth': 2, 'Dumbbell': 3, 'RemoteControl': 3, 'TableTopDecor': 1}
bathrooms = {'Towel': 30, 'HandTowel': 30, 'Plunger': 30, 'SoapBar': 30, 'SoapBottle': 30, 'Cloth': 30, 'PaperTowelRoll': 4, 'Candle': 30, 'SprayBottle': 30, 'ScrubBrush': 30, 'DishSponge': 6, 'TissueBox': 9, 'Footstool': 1}

kitchens_objects = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug", "Potato", "Pan", "Egg", "Spatula", "Cup", 'SoapBottle']
living_rooms_objects = ['Box', 'Laptop', 'CellPhone', 'CreditCard', 'AlarmClock', "RemoteControl", 'Pillow', 'KeyChain']# ,'Newspaper']
bedrooms_objects = ['Book', 'Laptop', 'CellPhone', 'AlarmClock', 'KeyChain']
bathrooms_objects = ['Towel', 'Plunger', 'SoapBar', 'SoapBottle', 'Cloth', 'Candle', 'SprayBottle', 'ScrubBrush']

FULL_LIST_OF_OBJECTS = {
    'kitchens': kitchens_objects,
    'living_rooms': living_rooms_objects,
    'bedrooms': bedrooms_objects,
    'bathrooms': bathrooms_objects,
}