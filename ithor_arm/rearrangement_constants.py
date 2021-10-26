# modified from lucaw: https://github.com/allenai/ai2thor-rearrangement/blob/main/rearrange/constants.py
import colorsys
import random

import numpy as np

random.seed(0)
MAX_HAND_METERS = 0.5
FOV = 90
STEP_SIZE = 0.25

# fmt: off
REARRANGE_SIM_OBJECTS = [
    # A
    "AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "ArmChair",
    "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin", "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box",
    # B
    "Bread", "BreadSliced", "ButterKnife",
    # C
    "Cabinet", "Candle", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop", "CreditCard",
    "Cup", "Curtains",
    # D
    "Desk", "DeskLamp", "Desktop", "DiningTable", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell",
    # E
    "Egg", "EggCracked",
    # F
    "Faucet", "Floor", "FloorLamp", "Footstool", "Fork", "Fridge",
    # G
    "GarbageBag", "GarbageCan",
    # H
    "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", "Knife",
    # L
    "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch",
    # M
    "Microwave", "Mirror", "Mug",
    # N
    "Newspaper",
    # O
    "Ottoman",
    # P
    "Painting", "Pan", "PaperTowel", "Pen", "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot",
    "Potato", "PotatoSliced",
    # R
    "RemoteControl", "RoomDecor",
    # S
    "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
    "ShowerHead", "SideTable", "Sink", "SinkBasin", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle",
    "Statue", "Stool", "StoveBurner", "StoveKnob",
    # T
    "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", "TissueBox", "Toaster", "Toilet",
    "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", "Towel", "TowelHolder", "TVStand",
    # V
    "VacuumCleaner", "Vase",
    # W
    "Watch", "WateringCan", "Window", "WineBottle",
]
# fmt: on

BIGGER = {
    "Book": {"openable": True, "receptacle": False, "pickupable": True},
    "Bread": {"openable": False, "receptacle": False, "pickupable": True},
    "Potato": {"openable": False, "receptacle": False, "pickupable": True},
    "SoapBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Kettle": {"openable": True, "receptacle": False, "pickupable": True},
    "Pan": {"openable": False, "receptacle": True, "pickupable": True},
    "Plate": {"openable": False, "receptacle": True, "pickupable": True},
    "Tomato": {"openable": False, "receptacle": False, "pickupable": True},
    "Vase": {"openable": False, "receptacle": False, "pickupable": True},
    "WineBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Pot": {"openable": False, "receptacle": True, "pickupable": True},
    "Lettuce": {"openable": False, "receptacle": False, "pickupable": True},
    "Statue": {"openable": False, "receptacle": False, "pickupable": True},
    "SprayBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Box": {"openable": True, "receptacle": True, "pickupable": True},
    "Laptop": {"openable": True, "receptacle": False, "pickupable": True},
    "Pillow": {"openable": False, "receptacle": False, "pickupable": True},
    "WateringCan": {"openable": False, "receptacle": False, "pickupable": True},
    "Boots": {"openable": False, "receptacle": False, "pickupable": True},
    "BasketBall": {"openable": False, "receptacle": False, "pickupable": True},
    "TennisRacket": {"openable": False, "receptacle": False, "pickupable": True},
    "TeddyBear": {"openable": False, "receptacle": False, "pickupable": True},
    "Cloth": {"openable": False, "receptacle": False, "pickupable": True},
    "Dumbbell": {"openable": False, "receptacle": False, "pickupable": True},
    "Footstool": {"openable": False, "receptacle": True, "pickupable": True},
    "Towel": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowel": {"openable": False, "receptacle": False, "pickupable": True},
    "Plunger": {"openable": False, "receptacle": False, "pickupable": True},
    "ToiletPaper": {"openable": False, "receptacle": False, "pickupable": True},
    "ScrubBrush": {"openable": False, "receptacle": False, "pickupable": True},
}
# fmt: off
OBJECT_TYPES_WITH_PROPERTIES = {
    "StoveBurner": {"openable": False, "receptacle": True, "pickupable": False},
    "Drawer": {"openable": True, "receptacle": True, "pickupable": False},
    "CounterTop": {"openable": False, "receptacle": True, "pickupable": False},
    "Cabinet": {"openable": True, "receptacle": True, "pickupable": False},
    "StoveKnob": {"openable": False, "receptacle": False, "pickupable": False},
    "Window": {"openable": False, "receptacle": False, "pickupable": False},
    "Sink": {"openable": False, "receptacle": True, "pickupable": False},
    "Floor": {"openable": False, "receptacle": True, "pickupable": False},
    "Book": {"openable": True, "receptacle": False, "pickupable": True},
    "Bottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Knife": {"openable": False, "receptacle": False, "pickupable": True},
    "Microwave": {"openable": True, "receptacle": True, "pickupable": False},
    "Bread": {"openable": False, "receptacle": False, "pickupable": True},
    "Fork": {"openable": False, "receptacle": False, "pickupable": True},
    "Shelf": {"openable": False, "receptacle": True, "pickupable": False},
    "Potato": {"openable": False, "receptacle": False, "pickupable": True},
    "HousePlant": {"openable": False, "receptacle": False, "pickupable": False},
    "Toaster": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Kettle": {"openable": True, "receptacle": False, "pickupable": True},
    "Pan": {"openable": False, "receptacle": True, "pickupable": True},
    "Plate": {"openable": False, "receptacle": True, "pickupable": True},
    "Tomato": {"openable": False, "receptacle": False, "pickupable": True},
    "Vase": {"openable": False, "receptacle": False, "pickupable": True},
    "GarbageCan": {"openable": False, "receptacle": True, "pickupable": False},
    "Egg": {"openable": False, "receptacle": False, "pickupable": True},
    "CreditCard": {"openable": False, "receptacle": False, "pickupable": True},
    "WineBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Pot": {"openable": False, "receptacle": True, "pickupable": True},
    "Spatula": {"openable": False, "receptacle": False, "pickupable": True},
    "PaperTowelRoll": {"openable": False, "receptacle": False, "pickupable": True},
    "Cup": {"openable": False, "receptacle": True, "pickupable": True},
    "Fridge": {"openable": True, "receptacle": True, "pickupable": False},
    "CoffeeMachine": {"openable": False, "receptacle": True, "pickupable": False},
    "Bowl": {"openable": False, "receptacle": True, "pickupable": True},
    "SinkBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "SaltShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "PepperShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "Lettuce": {"openable": False, "receptacle": False, "pickupable": True},
    "ButterKnife": {"openable": False, "receptacle": False, "pickupable": True},
    "Apple": {"openable": False, "receptacle": False, "pickupable": True},
    "DishSponge": {"openable": False, "receptacle": False, "pickupable": True},
    "Spoon": {"openable": False, "receptacle": False, "pickupable": True},
    "LightSwitch": {"openable": False, "receptacle": False, "pickupable": False},
    "Mug": {"openable": False, "receptacle": True, "pickupable": True},
    "ShelvingUnit": {"openable": False, "receptacle": True, "pickupable": False},
    "Statue": {"openable": False, "receptacle": False, "pickupable": True},
    "Stool": {"openable": False, "receptacle": True, "pickupable": False},
    "Faucet": {"openable": False, "receptacle": False, "pickupable": False},
    "Ladle": {"openable": False, "receptacle": False, "pickupable": True},
    "CellPhone": {"openable": False, "receptacle": False, "pickupable": True},
    "Chair": {"openable": False, "receptacle": True, "pickupable": False},
    "SideTable": {"openable": False, "receptacle": True, "pickupable": False},
    "DiningTable": {"openable": False, "receptacle": True, "pickupable": False},
    "Pen": {"openable": False, "receptacle": False, "pickupable": True},
    "SprayBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Curtains": {"openable": False, "receptacle": False, "pickupable": False},
    "Pencil": {"openable": False, "receptacle": False, "pickupable": True},
    "Blinds": {"openable": True, "receptacle": False, "pickupable": False},
    "GarbageBag": {"openable": False, "receptacle": False, "pickupable": False},
    "Safe": {"openable": True, "receptacle": True, "pickupable": False},
    "Painting": {"openable": False, "receptacle": False, "pickupable": False},
    "Box": {"openable": True, "receptacle": True, "pickupable": True},
    "Laptop": {"openable": True, "receptacle": False, "pickupable": True},
    "Television": {"openable": False, "receptacle": False, "pickupable": False},
    "TissueBox": {"openable": False, "receptacle": False, "pickupable": True},
    "KeyChain": {"openable": False, "receptacle": False, "pickupable": True},
    "FloorLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "DeskLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "Pillow": {"openable": False, "receptacle": False, "pickupable": True},
    "RemoteControl": {"openable": False, "receptacle": False, "pickupable": True},
    "Watch": {"openable": False, "receptacle": False, "pickupable": True},
    "Newspaper": {"openable": False, "receptacle": False, "pickupable": True},
    "ArmChair": {"openable": False, "receptacle": True, "pickupable": False},
    "CoffeeTable": {"openable": False, "receptacle": True, "pickupable": False},
    "TVStand": {"openable": False, "receptacle": True, "pickupable": False},
    "Sofa": {"openable": False, "receptacle": True, "pickupable": False},
    "WateringCan": {"openable": False, "receptacle": False, "pickupable": True},
    "Boots": {"openable": False, "receptacle": False, "pickupable": True},
    "Ottoman": {"openable": False, "receptacle": True, "pickupable": False},
    "Desk": {"openable": False, "receptacle": True, "pickupable": False},
    "Dresser": {"openable": False, "receptacle": True, "pickupable": False},
    "Mirror": {"openable": False, "receptacle": False, "pickupable": False},
    "DogBed": {"openable": False, "receptacle": True, "pickupable": False},
    "Candle": {"openable": False, "receptacle": False, "pickupable": True},
    "RoomDecor": {"openable": False, "receptacle": False, "pickupable": False},
    "Bed": {"openable": False, "receptacle": True, "pickupable": False},
    "BaseballBat": {"openable": False, "receptacle": False, "pickupable": True},
    "BasketBall": {"openable": False, "receptacle": False, "pickupable": True},
    "AlarmClock": {"openable": False, "receptacle": False, "pickupable": True},
    "CD": {"openable": False, "receptacle": False, "pickupable": True},
    "TennisRacket": {"openable": False, "receptacle": False, "pickupable": True},
    "TeddyBear": {"openable": False, "receptacle": False, "pickupable": True},
    "Poster": {"openable": False, "receptacle": False, "pickupable": False},
    "Cloth": {"openable": False, "receptacle": False, "pickupable": True},
    "Dumbbell": {"openable": False, "receptacle": False, "pickupable": True},
    "LaundryHamper": {"openable": True, "receptacle": True, "pickupable": False},
    "TableTopDecor": {"openable": False, "receptacle": False, "pickupable": True},
    "Desktop": {"openable": False, "receptacle": False, "pickupable": False},
    "Footstool": {"openable": False, "receptacle": True, "pickupable": True},
    "BathtubBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "ShowerCurtain": {"openable": True, "receptacle": False, "pickupable": False},
    "ShowerHead": {"openable": False, "receptacle": False, "pickupable": False},
    "Bathtub": {"openable": False, "receptacle": True, "pickupable": False},
    "Towel": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowel": {"openable": False, "receptacle": False, "pickupable": True},
    "Plunger": {"openable": False, "receptacle": False, "pickupable": True},
    "TowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ToiletPaperHanger": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBar": {"openable": False, "receptacle": False, "pickupable": True},
    "ToiletPaper": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ScrubBrush": {"openable": False, "receptacle": False, "pickupable": True},
    "Toilet": {"openable": True, "receptacle": True, "pickupable": False},
    "ShowerGlass": {"openable": False, "receptacle": False, "pickupable": False},
    "ShowerDoor": {"openable": True, "receptacle": False, "pickupable": False},
    "AluminumFoil": {"openable": False, "receptacle": False, "pickupable": True},
    "VacuumCleaner": {"openable": False, "receptacle": False, "pickupable": False}
}
# fmt: on


def _get_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    random.shuffle(colors)
    return colors


REARRANGE_SIM_OBJECT_COLORS = _get_colors(len(OBJECT_TYPES_WITH_PROPERTIES))
REARRANGE_SIM_OBJECTS_COLOR_LOOKUP = {
    p: REARRANGE_SIM_OBJECT_COLORS[i] for i, p in enumerate(OBJECT_TYPES_WITH_PROPERTIES)}

PICKUPABLE_OBJECTS = set(
    sorted(
        [
            object_type
            for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
            if properties["pickupable"]
        ]
    )
)

OPENABLE_OBJECTS = set(
    sorted(
        [
            object_type
            for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
            if properties["openable"] and not properties["pickupable"]
        ]
    )
)

RECEPTACLE_OBJECTS = set(
    sorted(
        [
            object_type
            for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
            if properties["receptacle"] and not properties["pickupable"]
        ]
    )
)

MAX_OPEN_RETRIES_REARRANGE = 10
MAX_MOVE_RETRIES_REARRANGE = 150

CONTROLLER_COMMIT_ID = "6f13532966080a051127167c6eb2117e47d96f3a"
# "62bba7e2537fb6aaf2ed19125b9508c8b99bced3"
ROOMR_CONTROLLER_COMMIT_ID = "f46d5ec42b65fdae9d9a48db2b4fb6d25afbd1fe"

OBJECT_TYPES_TO_NOT_MOVE = {
    "Apple",
    "Bread",
    "Cloth",
    "HandTowel",
    "KeyChain",
    "Lettuce",
    "Pillow",
    "Potato",
    "Tomato",
}
OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES = [
    "AluminumFoil",
    "CD",
    "Dumbbell",
    "Ladle",
    "Vase",
]

ACTION_NEGATIONS = {
    'MoveAhead': 'MoveBack',
    'MoveBack': 'MoveAhead',
    'RotateRight': 'RotateLeft',
    'RotateLeft': 'RotateRight',
    'LookDown' : 'LookUp',
    'LookUp' : 'LookDown',
    'MoveLeft' : 'MoveRight',
    'MoveRight' : 'MoveLeft'
}

#NOTE: below are constants from klemen, thanks klemen!

# OMNI_CATEGORIES is a list of all of the lvis categories plus all of the unique ithor categories
OMNI_CATEGORIES = [
    'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance',
    'amplifier', 'anklet', 'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium', 'arctic_(type_of_shoe)',
    'armband', 'armchair', 'armoire', 'armor', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado',
    'award', 'awning', 'ax', 'baboon', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
    'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna',
    'banjo', 'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat',
    'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe',
    'bathtub', 'batter_(food)', 'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedpan',
    'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle',
    'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
    'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card', 'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
    'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry', 'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
    'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase', 'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
    'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball',
    'box', 'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread', 'breechcloth', 'bridal_gown',
    'briefcase', 'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog',
    'bulldozer', 'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)',
    'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet', 'locker', 'cake', 'calculator',
    'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder',
    'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap', 'cape',
    'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan',
    'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast',
    'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue', 'chalice',
    'chandelier', 'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
    'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker', 'chopping_board',
    'chopstick', 'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent',
    'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin',
    'clutch_bag', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
    'coffeepot', 'coil', 'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book', 'compass', 'computer_keyboard',
    'condiment', 'cone', 'control', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie', 'cooking_utensil', 'cooler_(for_food)',
    'cork_(bottle_plug)', 'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset', 'costume', 'cougar',
    'coverall', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker', 'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
    'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)',
    'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain', 'cushion', 'cylinder',
    'cymbal', 'dagger', 'dalmatian', 'dartboard', 'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'diary',
    'die', 'dinghy', 'dining_table', 'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dishwasher_detergent', 'dispenser',
    'diving_board', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut',
    'dove', 'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drone', 'dropper',
    'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle',
    'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair',
    'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret', 'Ferris_wheel',
    'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose',
    'fireplace', 'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
    'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair',
    'food_processor', 'football_(American)', 'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener',
    'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle',
    'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)',
    'globe', 'glove', 'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater', 'gravestone',
    'gravy_boat', 'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
    'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel', 'handcart',
    'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband', 'headboard', 'headlight',
    'headscarf', 'headset', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus',
    'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah', 'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate',
    'hot_sauce', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear', 'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
    'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey', 'jet_plane',
    'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono', 'kitchen_sink', 'kitchen_table',
    'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'ladder', 'ladle',
    'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'lasagna', 'latch', 'lawn_mower',
    'leather', 'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
    'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'machine_gun',
    'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker',
    'martini', 'mascot', 'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup', 'measuring_stick',
    'meatball', 'medicine', 'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy',
    'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motor_vehicle',
    'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument',
    'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'nosebag_(for_animals)',
    'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
    'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad',
    'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose',
    'papaya', 'paper_plate', 'paper_towel', 'paperback_book', 'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parchment',
    'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport', 'pastry', 'patty_(food)', 'pea_(food)',
    'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg', 'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
    'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
    'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow', 'pin_(non_jewelry)',
    'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork',
    'pizza', 'place_mat', 'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
    'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
    'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune', 'pudding',
    'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit', 'race_car', 'racket',
    'radar', 'radiator', 'radio_receiver', 'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat', 'razorblade', 'reamer_(juicer)',
    'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat',
    'road_map', 'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade', 'rolling_pin', 'root_beer', 'router_(computer_equipment)', 'rubber_band',
    'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin', 'sail', 'salad', 'salad_plate', 'salami',
    'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
    'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver', 'scrubbing_brush',
    'sculpture', 'seabird', 'seahorse', 'seaplane', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)',
    'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants',
    'shot_glass', 'shoulder_bag', 'shovel', 'shower_head', 'shower_cap', 'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink', 'skateboard',
    'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)', 'slipper_(footwear)', 'smoothie',
    'snake', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball', 'solar_array', 'sombrero', 'soup', 'soup_bowl',
    'soupspoon', 'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
    'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)', 'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)',
    'steak_(food)', 'steak_knife', 'steering_wheel', 'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup', 'stool', 'stop_sign',
    'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer',
    'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
    'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table', 'table_lamp', 'tablecloth',
    'tachometer', 'taco', 'tag', 'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)',
    'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
    'telephone_pole', 'telephoto_lens', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'tequila', 'thermometer', 'thermos_bottle',
    'thermostat', 'thimble', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)',
    'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla',
    'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)',
    'trampoline', 'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk',
    'vat', 'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
    'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture', 'waffle',
    'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe', 'washbasin', 'automatic_washer',
    'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_heater', 'water_jug', 'water_gun', 'water_scooter', 'water_ski', 'water_tower',
    'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle',
    'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)',
    'wok', 'wolf', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini',
    'basket_ball', 'blinds', 'butter_knife', 'cd', 'cloth', 'counter_top', 'credit_card', 'curtains', 'house_plant', 'key_chain', 'clothes_hamper_lid',
    'light_switch', 'plunger', 'safe', 'shelf', 'shower_door', 'shower_glass', 'sink_basin', 'spray_bottle', 'stove_burner', 'stove_burner', 'side_table',
    'tissue_box', 'toilet_paper_hanger', 'television_set_stand', 'window', 'apple_sliced', 'tomato_sliced', 'lettuce_sliced', 'egg_cracked', 'bread_sliced',
    'potato_sliced']

# ITHOR_TO_OMNI is a dictionary mapping ithor class names to the omni category list
ITHOR_TO_OMNI = {
    "AlarmClock": "alarm_clock",
    "Apple": "apple",
    "ArmChair": "armchair",
    "BaseballBat": "baseball_bat",
    "Bathtub": "bathtub",
    "BathtubBasin": "bathtub",
    "BasketBall": "basket_ball",
    "Bed": "bed",
    "Blinds": "blinds",
    "Book": "book",
    "Boots": "boot",
    "ButterKnife": "butter_knife",
    "Bowl": "bowl",
    "Box": "box",
    "Bread": "bread",
    "Cabinet": "cabinet",
    "Candle": "candle",
    "Cart": "cart",
    "CellPhone": "cellular_telephone",
    "CoffeeMachine": "coffee_maker",
    "CounterTop": "counter_top",
    "Chair": "chair",
    "CD": "cd",
    "Cup": "cup",
    "Curtains": "curtains",
    "Cloth": "cloth",
    "CreditCard": "credit_card",
    "Desk": "desk",
    "DeskLamp": "table_lamp",
    "DishSponge": "sponge",
    "Drawer": "drawer",
    "Dresser": "dresser",
    "Egg": "egg",
    "Footstool": "footstool",
    "Fork": "fork",
    "FloorLamp": "lamp",
    "Fridge": "refrigerator",
    "GarbageCan": "trash_can",
    "Glassbottle": "bottle",
    "HandTowel": "hand_towel",
    "HandTowelHolder": "towel_rack",
    "HousePlant": "house_plant",
    "Kettle": "kettle",
    "Knife": "knife",
    "KeyChain": "key_chain",
    "Ladle": "ladle",
    "Lettuce": "lettuce",
    "Laptop": "laptop_computer",
    "LaundryHamper": "clothes_hamper",
    "LaundryHamperLid": "clothes_hamper_lid",
    "LightSwitch": "light_switch",
    "Mirror": "mirror",
    "Mug": "mug",
    "Microwave": "microwave_oven",
    "Newspaper": "newspaper",
    "Ottoman": "ottoman",
    "Painting": "painting",
    "PaperTowel": "paper_towel",
    "Pen": "pen",
    "Pencil": "pencil",
    "Pillow": "pillow",
    "Plate": "plate",
    "Poster": "poster",
    "Pot": "pot",
    "Pan": "frying_pan",
    "Potato": "potato",
    "PaperTowelRoll": "paper_towel",
    "PepperShaker": "pepper_mill",
    "Plunger": "plunger",
    "RemoteControl": "remote_control",
    "Sink": "sink",
    "SinkBasin": "sink_basin",
    "Sofa": "sofa",
    "Spatula": "spatula",
    "Spoon": "spoon",
    "Safe": "safe",
    "SoapBar": "soap",
    "SoapBottle": "soap",
    "SaltShaker": "saltshaker",
    "ScrubBrush": "scrubbing_brush",
    "Shelf": "shelf",
    "ShowerDoor": "shower_door",
    "ShowerGlass": "shower_glass",
    "SprayBottle": "spray_bottle",
    "Statue": "statue_(sculpture)",
    "StoveBurner": "stove_burner",
    "StoveKnob": "stove_burner",
    "SideTable": "side_table",
    "DiningTable": "dining_table",
    "CoffeeTable": "coffee_table",
    "TeddyBear": "teddy_bear",
    "TennisRacket": "tennis_racket",
    "Toaster": "toaster",
    "Toilet": "toilet",
    "Tomato": "tomato",
    "Towel": "towel",
    "Television": "television_set",
    "TissueBox": "tissue_box",
    "ToiletPaper": "toilet_tissue",
    "ToiletPaperRoll": "toilet_tissue",
    "ToiletPaperHanger": "toilet_paper_hanger",
    "TowelHolder": "towel_rack",
    "TVStand": "television_set_stand",
    "Vase": "vase",
    "Watch": "watch",
    "WateringCan": "watering_can",
    "WineBottle": "wine_bottle",
    "Window": "window",
    "ShowerCurtain": "shower_curtain",
    "Lamp": "lamp",
    "ShowerHead": "shower_head",
    "Faucet": "faucet",
    "AppleSliced": "apple_sliced",
    "TomatoSliced": "tomato_sliced",
    "LettuceSliced": "lettuce_sliced",
    "EggCracked": "egg_cracked",
    "BreadSliced": "bread_sliced",
    "PotatoSliced": "potato_sliced"
}

# OMNI_TO_ITHOR is the inverse mappingn of ITHOR_TO_OMNI
OMNI_TO_ITHOR = {v: k for k, v in ITHOR_TO_OMNI.items()}