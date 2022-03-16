

kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

all_scenes = kitchens + living_rooms + bedrooms + bathrooms

kitchens = {'Book': 2, 'Bottle': 7, 'Knife': 30, 'Bread': 30, 'Fork': 30, 'Potato': 30, 'SoapBottle': 30, 'Kettle': 15, 'Pan': 30, 'Plate': 30, 'Tomato': 30, 'Egg': 30, 'CreditCard': 3, 'WineBottle': 10, 'Pot': 30, 'Spatula': 30, 'PaperTowelRoll': 10, 'Cup': 30, 'Bowl': 30, 'SaltShaker': 30, 'PepperShaker': 30, 'Lettuce': 30, 'ButterKnife': 30, 'Apple': 30, 'DishSponge': 30, 'Spoon': 30, 'Mug': 30, 'Statue': 3, 'Ladle': 9, 'CellPhone': 4, 'Pen': 2, 'SprayBottle': 2, 'Pencil': 2}
living_rooms = {'Book': 7, 'Box': 30, 'Statue': 16, 'Laptop': 30, 'TissueBox': 9, 'CreditCard': 30, 'Plate': 10, 'KeyChain': 30, 'Vase': 7, 'Pencil': 4, 'Pillow': 30, 'Bowl': 5, 'RemoteControl': 30, 'Watch': 16, 'Pen': 5, 'Newspaper': 18, 'WateringCan': 13, 'Boots': 5, 'CellPhone': 7, 'Candle': 2}
bedrooms = {'Book': 30, 'Box': 12, 'Laptop': 30, 'CellPhone': 30, 'BaseballBat': 12, 'BasketBall': 11, 'TissueBox': 4, 'CreditCard': 30, 'AlarmClock': 30, 'Pencil': 30, 'Boots': 3, 'Pillow': 18, 'KeyChain': 30, 'Bowl': 13, 'Watch': 2, 'CD': 28, 'Pen': 30, 'Mug': 16, 'Statue': 4, 'TennisRacket': 10, 'TeddyBear': 10, 'Vase': 1, 'Cloth': 2, 'Dumbbell': 3, 'RemoteControl': 3, 'TableTopDecor': 1}
bathrooms = {'Towel': 30, 'HandTowel': 30, 'Plunger': 30, 'SoapBar': 30, 'SoapBottle': 30, 'Cloth': 30, 'PaperTowelRoll': 4, 'Candle': 30, 'SprayBottle': 30, 'ScrubBrush': 30, 'DishSponge': 6, 'TissueBox': 9, 'Footstool': 1}

kitchens_objects = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug", "Potato", "Pan", "Egg", "Spatula", "Cup", 'SoapBottle']
living_rooms_objects = ['Box', 'Laptop', 'CellPhone', 'CreditCard', 'AlarmClock', "RemoteControl", 'Pillow', 'KeyChain']# ,'Newspaper']
bedrooms_objects = ['Book', 'Laptop', 'CellPhone', 'AlarmClock', 'KeyChain']
bathrooms_objects = ['Towel', 'Plunger', 'SoapBar', 'SoapBottle', 'Cloth', 'Candle', 'SprayBottle', 'ScrubBrush']


KITCHEN_TRAIN = [f"FloorPlan{i}" for i in range(1, 21)]
KITCHEN_VAL = [f"FloorPlan{i}" for i in range(21, 26)]
KITCHEN_TEST = [f"FloorPlan{i}" for i in range(26, 31)]

LIVING_ROOM_TRAIN = [f"FloorPlan{200 + i}" for i in range(1, 21)]
LIVING_ROOM_VAL = [f"FloorPlan{200 + i}" for i in range(21, 26)]
LIVING_ROOM_TEST = [f"FloorPlan{200 + i}" for i in range(26, 31)]

BEDROOM_TRAIN = [f"FloorPlan{300 + i}" for i in range(1, 21)]
BEDROOM_VAL = [f"FloorPlan{300 + i}" for i in range(21, 26)]
BEDROOM_TEST = [f"FloorPlan{300 + i}" for i in range(26, 31)]

BATHROOM_TRAIN = [f"FloorPlan{400 + i}" for i in range(1, 21)]
BATHROOM_VAL = [f"FloorPlan{400 + i}" for i in range(21, 26)]
BATHROOM_TEST = [f"FloorPlan{400 + i}" for i in range(26, 31)]


ROBOTHOR_TRAIN= [f"FloorPlan_Train{i}_{j}" for i in range(1, 13) for j in range(1,6)]
ROBOTHOR_VAL= [f"FloorPlan_Val{i}_{j}" for i in range(1, 4) for j in range(1,6)]
ROBOTHOR_SCENE_NAMES = ROBOTHOR_TRAIN + ROBOTHOR_VAL
ROBOTHOR_MOVEABLE_UNIQUE_OBJECTS = ['Vase', 'Bowl', 'AlarmClock', 'Bottle', 'Mug', 'SprayBottle', 'BasketBall', 'RemoteControl', 'BaseballBat', 'Laptop', 'Apple', 'Box']

FULL_LIST_OF_OBJECTS = {
    'kitchens': kitchens_objects,
    'living_rooms': living_rooms_objects,
    'bedrooms': bedrooms_objects,
    'bathrooms': bathrooms_objects,
    'robothor': ROBOTHOR_MOVEABLE_UNIQUE_OBJECTS,
}

ROOM_TYPE_TO_IDS = {
    'kitchens': KITCHEN_TRAIN + KITCHEN_VAL + KITCHEN_TEST,
    'living_rooms': LIVING_ROOM_TRAIN + LIVING_ROOM_VAL + LIVING_ROOM_TEST,
    'bedrooms': BEDROOM_TRAIN + BEDROOM_VAL + BEDROOM_TEST,
    'bathrooms': BATHROOM_TRAIN + BATHROOM_VAL + BATHROOM_TEST,
    'robothor': ROBOTHOR_TRAIN + ROBOTHOR_VAL,
}
ROOM_ID_TO_TYPE = {r:r_type for r_type,r_list in ROOM_TYPE_TO_IDS.items() for r in r_list}

def get_room_type_from_id(room_id):
    if room_id in ROOM_ID_TO_TYPE:
        return ROOM_ID_TO_TYPE[room_id]
    elif room_id.replace('_physics', '') in ROOM_ID_TO_TYPE:
        return ROOM_ID_TO_TYPE[room_id.replace('_physics', '')]
    else:
        print('Room not found', room_id)
        return 'UNKNOWN'
# all_kithcens = [f"FloorPlan{i}" for i in range(1, 31)]
# all_living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
# all_bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
# all_bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]