import argparse
import datetime
import os
import pdb

import ai2thor.controller
import json

import cv2
import glob

# from helper_mover import reset_the_scene_and_get_reachables, initialize_arm, get_reachable_positions, only_reset_scene, transport_wrapper, is_object_in_receptacle
# from helper_mover import ENV_ARGS
# from utils.ideal_pose_util import SCENE_START_CHEATING_LOCATIONS
# from utils.possible_agent_location_util import transport_agent_to_closest_object, prune_countertops
#
# from utils.possible_object_location_utils import get_all_counter_tops_possible_locations
from scripts.dataset_generation.util_agent_location import prune_countertops, transport_agent_to_closest_object
from scripts.dataset_generation.util_find_object_possible_location import get_all_counter_tops_possible_locations
from scripts.stretch_jupyter_helper import reset_environment_and_additional_commands, get_reachable_positions, transport_wrapper, is_object_in_receptacle
from utils.stretch_utils.stretch_constants import STRETCH_MANIPULATHOR_COMMIT_ID, STRETCH_ENV_ARGS

screen_size=224

STRETCH_ENV_ARGS['width'] = screen_size
STRETCH_ENV_ARGS['height'] = screen_size
STRETCH_ENV_ARGS['commit_id']=STRETCH_MANIPULATHOR_COMMIT_ID
STRETCH_ENV_ARGS['renderDepthImage'] = False
STRETCH_ENV_ARGS['renderInstanceSegmentation'] = False


def round_float(number, round_digit=2):
    return round(number *100) / 100

def parse_args():
    parser = argparse.ArgumentParser(description='Data loader')
    parser.add_argument('--scene_name', default='FloorPlan1_physics')
    parser.add_argument('--object_type', default='Apple') #['Apple', 'Bread', 'Book', 'Tomato', 'Lettuce', 'Pot', 'Mug']
    # parser.add_argument('--screen_size', default=224, type=int)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--redo', action='store_true', default=False)
    parser.add_argument('--countertop', default=None)


    args = parser.parse_args()
    time_to_write = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    args.result_adr = 'apnd-dataset/valid_object_positions/valid_{}_positions_in_{}.json'.format(args.object_type, args.scene_name)
    os.makedirs('apnd-dataset/valid_object_positions', exist_ok=True)
    processed_before = [f for f in glob.glob(args.result_adr.split('.json')[0] + '*')]
    args.processed_before = processed_before
    if args.redo:
        args.processed_before = []
    if args.visualize:
        args.visualize_adr = 'saved_images/{}_{}_{}'.format(time_to_write, args.scene_name, args.object_type)
        os.makedirs(args.visualize_adr)
    return args

def main():

    args = parse_args()

    scene_name = args.scene_name
    object_type = args.object_type
    result_adr = args.result_adr

    if args.processed_before:
        print('This is already done', scene_name, object_type)
        return
    controller = ai2thor.controller.Controller(
        **STRETCH_ENV_ARGS
    )
    reset_environment_and_additional_commands(controller, scene_name)

    target_obj = [o for o in controller.last_event.metadata['objects'] if o['objectType'] == object_type]
    if len(target_obj) == 0:
        print(object_type, 'not found in ', scene_name)
        print('Abort')
        return
    assert len(target_obj) == 1
    target_obj = target_obj[0]
    object_id = target_obj['objectId']

    #Get all the counter tops and locations on top of them
    all_countertops = get_all_counter_tops_possible_locations(controller)

    # prune countertops grids
    all_countertops = prune_countertops(all_countertops)

    # This block is not needed
    # #Find all the possible locations for this specific object on those countertops, far enough objects
    # pruned_for_this_object_countertop_locations = get_realistic_positions_for_object(all_countertops, object_id, controller)

    reachable_positions = get_reachable_positions(controller)

    full_dataset = []


    print('Total number of countertops', len(all_countertops))
    # TODO maybe it is easier to find reachable position for each countertop and only search among them?
    for i, countertop in enumerate(all_countertops):
        countertop_id, counterType, possible_positions = countertop['countertop_id'], countertop['counterType'], countertop['possible_positions']
        print('index', i, ': countertop', counterType, countertop_id)
        if counterType in ['Floor', 'GarbageCan', 'Fridge', 'Microwave']:
            continue

        if args.countertop is not None:
            if args.countertop not in counterType:
                continue
        this_countertop_total = len(possible_positions)
        this_countertop_visible = 0
        this_countertop_invisible = 0

        # prev_saved_position = None
        for (ind_position, target_position) in enumerate(possible_positions):

            reset_environment_and_additional_commands(controller, scene_name)
            # maybe reset the scene each time before calculating another possible location? that's too much though
            # consider using objectteleport, becareful object might clip https://github.com/allenai/ai2thor/blob/a23fcdf86f131e24a88ccc30aa1ca80e5ba7af0b/unity/Assets/Scripts/PhysicsRemoteFPSAgentController.cs#L709
            event , action_detail_list= transport_wrapper(controller, object_id, target_position)
            object_parent = event.get_object(object_id)['parentReceptacles']
            if event.metadata['lastActionSuccess'] == False:
                # print('transport failed')  remove this?
                continue
            elif not is_object_in_receptacle(controller.last_event,object_id,countertop_id): #if object was not moved or it has fallen over from this countrerop
                # print('object not in good receptacle') remove this?
                continue

            move_to_visible, agent_pose = transport_agent_to_closest_object(controller, object_id, reachable_positions, countertop_id)
            if not move_to_visible:
                this_countertop_invisible +=1
                # print('Oh No could not transport to somewhere visible', countertop_id)
                # we should still save these cases though
                full_dataset.append(dict(
                    object_id=object_id,
                    object_location=target_position,
                    scene_name=scene_name,
                    countertop_id=countertop_id,
                    agent_pose=agent_pose,
                    visibility=False,
                ))
            else:
                # # if prev_saved_position is too close to this one, ignore
                # current_position = target_position
                # if prev_saved_position is not None:
                #     distance = position_distance(prev_saved_position, current_position)
                #
                #     if distance < DIST_THR:
                #         if args.verbose:
                #             print('too close', prev_saved_position, current_position)
                #         continue

                # # set the prev_saved_position to the current one
                # prev_saved_position = current_position

                # if args.verbose:
                #     print('Adding ', object_id, target_position)
                this_countertop_visible += 1
                full_dataset.append(dict(
                    object_id=object_id,
                    object_location=target_position,
                    scene_name=scene_name,
                    countertop_id=countertop_id,
                    agent_pose=agent_pose,
                    visibility=True,
                ))

            if args.visualize:
                image = controller.last_event.frame
                image_adr = os.path.join(args.visualize_adr, '{}_{}_{}_visibility_{}.jpg'.format(countertop_id,str(ind_position), str(target_position), move_to_visible))
                cv2.imwrite(image_adr, image[:,:,[2,1,0]])


        this_countertop_success = this_countertop_invisible + this_countertop_visible + 1e-5
        this_countertop_total += 1e-5
        print('Total', len(possible_positions), 'visible', this_countertop_visible, 'PERCENT visible', round_float(this_countertop_visible / this_countertop_success), 'invisible', round_float(this_countertop_invisible / this_countertop_success), 'failed', round_float(1 - (this_countertop_invisible + this_countertop_visible) / this_countertop_total))

    print('Saved in ', result_adr)
    with open(result_adr, 'w') as f:
        json.dump({scene_name: full_dataset}, f)


if __name__ == '__main__':
    # debug()
    main()
