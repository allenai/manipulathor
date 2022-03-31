import io

import imageio
import torch
import os
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

from manipulathor_utils.debugger_util import ForkedPdb
from utils.stretch_utils.stretch_constants import MOVE_AHEAD, ROTATE_LEFT ,ROTATE_RIGHT ,MOVE_ARM_HEIGHT_P ,MOVE_ARM_HEIGHT_M ,MOVE_ARM_X_P ,MOVE_ARM_X_M ,MOVE_ARM_Y_P ,MOVE_ARM_Y_M ,MOVE_ARM_Z_P ,MOVE_ARM_Z_M ,PICKUP ,DONE, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, ROTATE_LEFT_SMALL, ROTATE_RIGHT_SMALL, MOVE_WRIST_P_SMALL, MOVE_WRIST_M_SMALL


def hacky_visualization(observations, object_mask, base_directory_to_right_images, query_objects=None, gt_mask=None, text_to_write=None, distance_vector_to_viz=None):
    def unnormalize_image(img):
        # img = img.squeeze(0).squeeze(0)
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
        viz_image = unnormalize_image(viz_image)
        def normalize_depth(depth):
            depth = depth.squeeze(0).squeeze(0)
            depth = depth.clamp(0,10) / 10
            depth = depth.repeat(1, 1, 3)
            return depth
        depth = normalize_depth(depth)

        list_of_visualizations = [viz_image, depth]

        if 'rgb_lowres_arm' in observations:
            kinect_image = observations['rgb_lowres_arm'].squeeze(0).squeeze(0)
            kinect_image = unnormalize_image(kinect_image)
            list_of_visualizations.append(kinect_image)
        if 'depth_lowres_arm' in observations:
            arm_depth = normalize_depth(observations['depth_lowres_arm'])
            list_of_visualizations.append(arm_depth)

        if query_objects is not None:
            viz_query_obj = query_objects.squeeze(0).squeeze(0).permute(1,2,0) #TO make it channel last
            viz_query_obj = unnormalize_image(viz_query_obj)
            list_of_visualizations.append(viz_query_obj)

        viz_mask = predicted_masks.squeeze(0).squeeze(0).repeat(1,1, 3)
        if text_to_write is not None:

            text_to_write = text_to_write.squeeze(0).squeeze(0)
            text_to_write = text_to_write * 10
            text_to_write = text_to_write.int()
            text_to_write = str(text_to_write.tolist()) + '=' + str(text_to_write.float().norm().item())
            viz_mask = put_additional_text_on_image([viz_mask], [text_to_write], color=(255,255,255))[0]


        list_of_visualizations.append(viz_mask)
        if gt_mask is not None:
            gt_mask = gt_mask.squeeze(0).squeeze(0).repeat(1,1, 3)
            list_of_visualizations.append(gt_mask)
        if distance_vector_to_viz is not None:

            arm_dist = distance_vector_to_viz['arm_dist'].squeeze(0).squeeze(0)
            agent_dist = distance_vector_to_viz['agent_dist'].squeeze(0).squeeze(0)

            viz_vector_dist = get_distance_vector_visualization_distances(arm_dist, agent_dist)

            # viz_vector_dist = fig2data(fig)
            list_of_visualizations.append(viz_vector_dist)
        combined = torch.cat(list_of_visualizations, dim=1)
        directory_to_write_images = os.path.join('experiment_output/visualizations_masks', base_directory_to_right_images)
        os.makedirs(directory_to_write_images, exist_ok=True)
        time_to_write = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.png")
        cv2.imwrite(os.path.join(directory_to_write_images, time_to_write), (combined[:,:,[2,1,0]] * 255.).int().cpu().numpy())

def get_distance_vector_visualization_distances(arm_dist, agent_dist):
    plt.cla()
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(224*px,224*px))
    ax = fig.add_subplot(111)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    # plt.plot([0, 0], [arm_dist[0], arm_dist[2]],linestyle='solid', label='arm',linewidth=3, color='blue')
    if arm_dist.sum() != 12:
        ax.arrow(0, 0, arm_dist[0], arm_dist[2],color='blue')#,head_width=3)
    if agent_dist.sum() != 12:
        ax.arrow(0, 0, agent_dist[0], agent_dist[2],color='green')#,head_width=3)
    viz_vector_dist = torch.Tensor(get_img_from_fig(fig).astype(np.float) / 255.)
    return viz_vector_dist
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

def get_img_from_fig(fig, dpi=180, w=224,h=224):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w,h))

    # fig.canvas.draw()
    # # grab the pixel buffer and dump it into a numpy array
    # img = np.array(fig.canvas.renderer.buffer_rgba())[:,:,3]
    return img
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
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

def save_image_list_to_gif(image_list, gif_name, gif_dir):
    gif_adr = os.path.join(gif_dir, gif_name)

    seq_len, cols, w, h, c = image_list.shape

    pallet = np.zeros((seq_len, w, h * cols, c))

    for col_ind in range(cols):
        pallet[:, :, col_ind * h : (col_ind + 1) * h, :] = image_list[:, col_ind]

    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    imageio.mimsave(gif_adr, pallet.astype(np.uint8), format="GIF", duration=1 / 5)
    print("Saved result in ", gif_adr)

def put_action_on_image(images, actions):
    all_images = []
    for i in range(len(images) - 1):
        img = images[i]
        action = actions[i]
        action_names = (MOVE_AHEAD,ROTATE_LEFT ,ROTATE_RIGHT ,MOVE_ARM_HEIGHT_P ,MOVE_ARM_HEIGHT_M ,MOVE_ARM_X_P ,MOVE_ARM_X_M ,MOVE_ARM_Y_P ,MOVE_ARM_Y_M ,MOVE_ARM_Z_P ,MOVE_ARM_Z_M ,PICKUP ,DONE, MOVE_BACK, MOVE_WRIST_P, MOVE_WRIST_M, ROTATE_LEFT_SMALL, ROTATE_RIGHT_SMALL, MOVE_WRIST_P_SMALL, MOVE_WRIST_M_SMALL)
        action_short = ("MOVE_AHEAD","ROTATE_L" ,"ROTATE_R" ,"ARM_H_P" ,"ARM_H_M" ,"ARM_X_P" ,"ARM_X_M" ,"ARM_Y_P" ,"ARM_Y_M" ,"ARM_Z_P" ,"ARM_Z_M" ,"PICKUP" ,"DONE", "MOVE_BACK", "WRIST_P", "WRIST_M", "ROTATE_L_S" ,"ROTATE_R_S" , "WRIST_P_S", "WRIST_M_S")
        action = action_short[action_names.index(action)]
        position = (10,10)

        from PIL import Image, ImageFont, ImageDraw
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, action, (0,0,0))
        all_images.append(np.array(pil_img))


    all_images.append(images[-1]) # No action needs to be written here
    return all_images


def put_additional_text_on_image(images, added_texts, color = (0,0,0)):
    all_images = []
    length_of_list = len(images)

    for i in range(length_of_list):
        if i == length_of_list - 1 and len(added_texts) < length_of_list:
            assert len(added_texts) == length_of_list - 1
            all_images.append(images[i]) # No action needs to be written here
            continue

        original_image = images[i]
        if type(original_image) == torch.Tensor:
            images[i] = (original_image* 255.).int().cpu().numpy().astype(np.uint8)
        img = images[i]
        text = added_texts[i]

        position = (10,200)
        from PIL import Image, ImageFont, ImageDraw
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, color)
        numpy_image = np.array(pil_img)

        if type(original_image) == torch.Tensor:
            numpy_image = torch.Tensor(numpy_image).float() / 255.
        all_images.append(numpy_image)



    return all_images

def depth_to_rgb(frame):
    frame = frame / 10 * 255.
    frame = frame.astype(np.uint8)
    frame = np.expand_dims(frame, axis=-1)
    frame = np.tile(frame,(1,1,3))
    return frame