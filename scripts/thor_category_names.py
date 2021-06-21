# import ai2thor.controller; c=ai2thor.controller.Controller(); thor_objects = set([o['objectType'] for o in c.last_event.metadata['objects']])
# {name:re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower().replace(' ', '_').replace('__', '_') for name in thor_objects}
# [k for k in lvis if 'plant' in k]
import pdb

from scripts.lvis_category_name import lvis_category_id2name

thor_possible_alternatives = {
    'Sink': ['sink', 'kitchen_sink'],
    'Shelf': ['shelf'],
    'StoveKnob': ['stove_knob'],
    'DishSponge': ['dish_sponge', 'sponge'],
    'Pan': ['pan', 'frying_pan', 'pan_(for_cooking)', 'saucepan'],
    'Apple': ['apple'],
    'Cup': ['cup', 'teacup'],
    'Bottle': ['bottle', 'beer_bottle', 'thermos_bottle', 'water_bottle', 'wine_bottle'],
    'Pot': ['pot', 'crock_pot'],
    # 'ShelvingUnit': ['shelving_unit', 'shelf'], #we can't have them both, choose your poison
    'StoveBurner': ['stove_burner', 'stove'],
    'Microwave': ['microwave', 'microwave_oven'],
    'CellPhone': ['cell_phone', 'cellular_telephone', 'telephone'],
    'Chair': ['chair', 'deck_chair', 'folding_chair', 'highchair'],
    'Spoon': ['soupspoon', 'spoon', 'wooden_spoon'],
    'Faucet': ['faucet', 'water_faucet'],
    'Toaster': ['toaster', 'toaster_oven'],
    'Plate': ['plate', 'paper_plate', 'salad_plate'],
    'Drawer': ['drawer', 'chest_of_drawers_(furniture)'],
    'Vase': ['vase'],
    'CoffeeMachine': ['coffee_machine', 'coffee_maker'],
    'Tomato': ['tomato'],
    'PepperShaker': ['pepper_shaker', 'pepper_mill'],
    'LightSwitch': ['light_switch'],
    'Egg': ['egg', 'boiled_egg', 'scrambled_eggs'],
    'PaperTowelRoll': ['paper_towel_roll', 'paper_towel', 'tissue_paper'],
    'Bowl': ['bowl', 'soup_bowl', 'sugar_bowl'],
    'Mug': ['mug'],
    'GarbageCan': ['garbage_can', 'trash_can'],
    'CounterTop': ['counter_top'],
    # 'ButterKnife': ['butter_knife', 'knife', 'pocketknife', 'steak_knife'], #we can't have them both, choose your poison
    # 'SinkBasin': ['sink_basin', 'sink'], #we can't have them both, choose your poison
    'Fridge': ['fridge', 'refrigerator'],
    'Floor': ['floor'],
    'SoapBottle': ['soap_bottle', 'soap'],
    'Window': ['window'],
    'Potato': ['potato', 'sweet_potato'],
    'Bread': ['bread', 'biscuit_(bread)', 'cornbread', 'pita_(bread)'],
    'Spatula': ['spatula'],
    'Knife': ['knife', 'pocketknife', 'steak_knife'],
    'CreditCard': ['credit_card', 'business_card'],
    'Cabinet': ['cabinet', 'file_cabinet'],
    'Lettuce': ['lettuce', 'romaine_lettuce'],
    'Statue': ['statue', 'statue_(sculpture)', 'sculpture'],
    'SaltShaker': ['salt_shaker', 'saltshaker'],
    'Fork': ['fork'],
    'HousePlant': ['house_plant', 'plant']
}
thor_possible_objects = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug", "Potato", "Pan", "Egg", "Spatula", "Cup"]
lvis_to_thor_translator = {}
for k, assigned in thor_possible_alternatives.items():
    for a in assigned:
        if a in lvis_category_id2name:
            assert a not in lvis_to_thor_translator, pdb.set_trace()
            lvis_to_thor_translator[a] = k

thor_object_name_to_lvis_valid_indices = {
    k: [lvis_category_id2name.index(m) for m in possibles if m in lvis_category_id2name]
    for (k, possibles) in thor_possible_alternatives.items()
}

