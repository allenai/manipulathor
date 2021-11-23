import torch
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import ImageFont
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from manipulathor_utils.debugger_util import ForkedPdb


def hacky_visualization(observations, object_mask, query_objects, base_directory_to_right_images, gt_mask=None, visual_compass_source=None, visual_compass_destination=None):
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
        viz_mask = overlay_mask(viz_image, viz_mask)
        list_of_visualizations = [add_text(viz_image, 'RGB'), add_text(depth, 'Depth'), add_text(viz_mask, 'Predicted Mask')] #TODO , viz_query_obj]
        if gt_mask is not None:
            gt_mask = gt_mask.squeeze(0).squeeze(0).repeat(1,1, 3)
            gt_mask = overlay_mask(viz_image, gt_mask)
            list_of_visualizations.append(add_text(gt_mask, 'GT Mask'))
        if visual_compass_source is not None:

            visual_compass_source = visual_compass_source.squeeze(0).squeeze(0)
            visual_compass_destination = visual_compass_destination.squeeze(0).squeeze(0)
            visualize_compass = visualize_3d_vector(visual_compass_source, visual_compass_destination)[:,:,:3]
            visualize_compass = torch.Tensor(visualize_compass) / 255.
            # ForkedPdb().set_trace()
            # import matplotlib.pyplot as plt; plt.imsave('something.png', visualize_3d_vector(visual_compass)[:,:,:3])
            list_of_visualizations.append(add_text(visualize_compass, 'Visual Compass'))
        combined = torch.cat(list_of_visualizations, dim=1)
        directory_to_write_images = os.path.join('experiment_output/visualizations_masks', base_directory_to_right_images)
        os.makedirs(directory_to_write_images, exist_ok=True)
        time_to_write = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.png")
        cv2.imwrite(os.path.join(directory_to_write_images, time_to_write), (combined[:,:,[2,1,0]] * 255.).int().cpu().numpy())
        # for i in range(len(list_of_visualizations)):
        #     cv2.imwrite(os.path.join(directory_to_write_images, time_to_write + f'_{i}.png'), (list_of_visualizations[i][:,:,[2,1,0]] * 255.).int().cpu().numpy())
        if 'topdown_view' in observations:
            combined = observations['topdown_view'].squeeze(0).squeeze(0)
            directory_to_write_images = os.path.join('experiment_output/visualizations_topdown', base_directory_to_right_images)
            os.makedirs(directory_to_write_images, exist_ok=True)
            time_to_write = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f.png")
            cv2.imwrite(os.path.join(directory_to_write_images, time_to_write), (combined[:,:,[2,1,0]] ).int().cpu().numpy())

def overlay_mask(image, mask):
    mask = mask[:,:,0]
    image = image.mean(-1).unsqueeze(-1).repeat(1,1,3)
    if mask.sum() > 0:
        thing_to_change = image[mask == 1]
        thing_to_change[:,0] += 0.2
        image[mask == 1] = thing_to_change
    image = image.clip(0,1)
    return image
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
def add_text(image, text):
    image = image.clone()
    # font = cv2.FONT_HERSHEY_SIMPLEX# org
    font = cv2.FONT_HERSHEY_PLAIN
    org = (10,20)# fontScale
    fontScale = 1# Red color in BGR
    if text == 'Visual Compass':
        color = (0,0,0)# Line thickness of 2 px
    else:
        color = (1,1,1)# Line thickness of 2 px
    thickness = 1
    result = cv2.putText(image.numpy(), text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
    return torch.Tensor(result)
def visualize_3d_vector(visual_compass_source, visual_compass_destination):
    def normalize(visual_compass):
        if visual_compass.sum() == 12:
            visual_compass[:]=0
        else:
            visual_compass = visual_compass / (visual_compass.norm() + 1e-9)
            visual_compass = - visual_compass
        return visual_compass
    visual_compass_source = normalize(visual_compass_source)
    visual_compass_destination = normalize(visual_compass_destination)
    # visual_compass = normalize(visual_compass)
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (2.24,2.24)

    plt.cla()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.view_init(30,60)
    ax.view_init(30,60+ 180)
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    def draw_arrow(visual_compass, color):
        x, y, z = [[0, l] for l in visual_compass]
        a = Arrow3D([x[0], x[1]],
                    [z[0], z[1]], [y[0],y[1]], mutation_scale=20,
                    lw=3, arrowstyle="-|>", color=color)
        ax.add_artist(a)
    draw_arrow(visual_compass_source, 'red')
    draw_arrow(visual_compass_destination, 'blue')
    result = fig2data(fig)
    plt.close(fig)
    #TODO add rgb depth
    return result
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

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