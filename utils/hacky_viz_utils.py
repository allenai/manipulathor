import torch
import os
from datetime import datetime
import cv2

from manipulathor_utils.debugger_util import ForkedPdb


def hacky_visualization(observations, object_mask, query_objects, base_directory_to_right_images, gt_mask=None):
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
        if 'colorful_depth' in observations:
            depth = observations['colorful_depth'].squeeze(0).squeeze(0)
        viz_query_obj = query_objects.squeeze(0).squeeze(0).permute(1,2,0) #TO make it channel last
        viz_mask = predicted_masks.squeeze(0).squeeze(0).repeat(1,1, 3)
        viz_image = unnormalize_image(viz_image)
        viz_query_obj = unnormalize_image(viz_query_obj)
        list_of_visualizations = [viz_image, depth, viz_mask, viz_query_obj]
        if gt_mask is not None:
            gt_mask = gt_mask.squeeze(0).squeeze(0).repeat(1,1, 3)
            list_of_visualizations.append(gt_mask)
        combined = torch.cat(list_of_visualizations, dim=1)
        directory_to_write_images = os.path.join('experiment_output/visualizations_masks', base_directory_to_right_images)
        os.makedirs(directory_to_write_images, exist_ok=True)
        time_to_write = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.png")
        cv2.imwrite(os.path.join(directory_to_write_images, time_to_write), (combined[:,:,[2,1,0]] * 255.).int().cpu().numpy())
        for i in range(len(list_of_visualizations)):
            cv2.imwrite(os.path.join(directory_to_write_images, time_to_write + f'_{i}.png'), (list_of_visualizations[i][:,:,[2,1,0]] * 255.).int().cpu().numpy())
        if 'topdown_view' in observations:
            combined = observations['topdown_view'].squeeze(0).squeeze(0)
            directory_to_write_images = os.path.join('experiment_output/visualizations_topdown', base_directory_to_right_images)
            os.makedirs(directory_to_write_images, exist_ok=True)
            time_to_write = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.png")
            cv2.imwrite(os.path.join(directory_to_write_images, time_to_write), (combined[:,:,[2,1,0]] ).int().cpu().numpy())


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