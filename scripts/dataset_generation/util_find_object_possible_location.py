def get_all_counter_tops_possible_locations(controller):
    event = controller.last_event
    all_receptacles = [o for o in event.metadata['objects'] if o['receptacle'] == True and not o['openable']]
    result = []
    for o in all_receptacles:

        countertop_id = o['objectId']
        event = controller.step('GetSpawnCoordinatesAboveReceptacle', objectId=countertop_id, anywhere=True)
        possible_xyz = event.metadata['actionReturn']
        current_positions = {'countertop_id': countertop_id, 'counterType':o['objectType'], 'possible_positions':possible_xyz}
        result.append(current_positions)
    return result

def distance(p1, p2):
    return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2) ** .5

def get_object_info(controller, object_id):
    return [o for o in controller.last_event.metadata['objects'] if o['objectId'] == object_id][0]

def get_object_location(controller, object_id):
    all_obj = controller.last_event.metadata['objects']
    target_obj = [o for o in all_obj if o['objectId'] == object_id]
    assert len(target_obj) == 1
    target_obj = target_obj[0]
    return target_obj
