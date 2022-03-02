import torch
import os
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ithor_arm.ithor_arm_viz import put_additional_text_on_image
from manipulathor_utils.debugger_util import ForkedPdb


def hacky_visualization(observations, object_mask, query_objects, base_directory_to_right_images, gt_mask=None, text_to_write=None):
    def unnormalize_image(img):
        mean=torch.Tensor([0.485, 0.456, 0.406]).to(img.device)
        std=torch.Tensor([0.229, 0.224, 0.225]).to(img.device)
        img = (img * std + mean)
        img = torch.clamp(img, 0, 1)
        return img
    depth = observations['depth_lowres']
    if 'only_detection_rgb_lowres' in observations:
        viz_image = observations['only_detection_rgb_lowres']
    elif 'rgb_lowres' in observations:
        viz_image = observations['rgb_lowres']
    else:
        viz_image = depth
    predicted_masks = object_mask
    bsize, seqlen, w, h, c = viz_image.shape
    if bsize == 1 and seqlen == 1:
        viz_image = viz_image.squeeze(0).squeeze(0)
        depth = depth.squeeze(0).squeeze(0)
        depth = depth.clamp(0,10) / 10
        depth = depth.repeat(1, 1, 3)

        viz_query_obj = query_objects.squeeze(0).squeeze(0).permute(1,2,0) #TO make it channel last
        viz_mask = predicted_masks.squeeze(0).squeeze(0).repeat(1,1, 3)
        if text_to_write is not None:

            text_to_write = text_to_write.squeeze(0).squeeze(0)
            text_to_write = text_to_write * 10
            text_to_write = text_to_write.int()
            text_to_write = str(text_to_write.tolist()) + '=' + str(text_to_write.float().norm().item())
            viz_mask = put_additional_text_on_image([viz_mask], [text_to_write], color=(255,255,255))[0]

        viz_image = unnormalize_image(viz_image)
        viz_query_obj = unnormalize_image(viz_query_obj)
        list_of_visualizations = [viz_image, depth, viz_query_obj, viz_mask]
        if gt_mask is not None:
            gt_mask = gt_mask.squeeze(0).squeeze(0).repeat(1,1, 3)
            list_of_visualizations.append(gt_mask)
        combined = torch.cat(list_of_visualizations, dim=1)
        directory_to_write_images = os.path.join('experiment_output/visualizations_masks', base_directory_to_right_images)
        os.makedirs(directory_to_write_images, exist_ok=True)
        time_to_write = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.png")
        cv2.imwrite(os.path.join(directory_to_write_images, time_to_write), (combined[:,:,[2,1,0]] * 255.).int().cpu().numpy())

def calc_dict_average(nested_dict):
    if type(nested_dict) == list:
        total_str = sum(nested_dict) / len(nested_dict)
    elif type(nested_dict) == float or type(nested_dict) == int:
        total_str = str(nested_dict)
    elif type(nested_dict) == dict:
        total_str = '{'
        for key, val in nested_dict.items():
            total_str += f'{key}:{calc_dict_average(val)},'
        total_str += '}'
    return total_str

def save_quick_frame(controller, image_adr, top_view=False):
    first_camera = controller.last_event.frame
    arm_camera = controller.last_event.third_party_camera_frames[0]
    if top_view:
        third_view = get_stretch_top_view(controller)
        combined_camera = np.concatenate([first_camera, arm_camera, third_view], axis=1)
    else:
        combined_camera = np.concatenate([first_camera, arm_camera], axis=1)
    plt.imsave(image_adr, combined_camera)

def get_stretch_top_view(controller): # TODO this might mess up some other things be careful
    camera_position = {
        'position': controller.last_event.metadata['cameraPosition'],
        'rotation': dict(x=90,y=0,z=0)
    }
    camera_position['position']['y'] += 0.5
    if len(controller.last_event.third_party_camera_frames) > 1:
        controller.step('UpdateThirdPartyCamera',
            thirdPartyCameraId=1, # id is available in the metadata response
            rotation=camera_position['rotation'],
            position=camera_position['position']
            )
    else:
        controller.step('AddThirdPartyCamera',
            rotation=camera_position['rotation'],
            position=camera_position['position'],
            fieldOfView=100)
    return( controller.last_event.third_party_camera_frames[-1])