"""Utility classes and functions for sensory inputs used by the models."""
import datetime
import glob
import os
import platform
import random
import time
from typing import Any, Union, Optional

import cv2
import gym
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
# from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from torch.distributions.utils import lazy_property

from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_sensors import DepthSensorThor
from manipulathor_utils.debugger_util import ForkedPdb
#
from utils.calculation_utils import calc_world_coordinates
from utils.detection_translator_util import THOR2COCO
from utils.noise_in_motion_util import squeeze_bool_mask
from utils.real_stretch_utils import get_binary_mask_of_arm, get_mid_point_of_object_from_depth_and_mask
from utils.stretch_utils.stretch_constants import INTEL_RESIZED_H, INTEL_RESIZED_W, KINECT_REAL_W, KINECT_REAL_H, \
    MAX_INTEL_DEPTH, MIN_INTEL_DEPTH, INTEL_FOV_W, INTEL_FOV_H, KINECT_FOV_W, \
    KINECT_FOV_H
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment


class RealRGBSensorStretchNav(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running StretchManipulaTHOREnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(
            self, env: StretchManipulaTHOREnvironment, task: Task[StretchManipulaTHOREnvironment]
    ) -> np.ndarray:  # type:ignore
        rgb = env.nav_rgb.copy()
        return (rgb)

class RealDepthSensorStretchNav(
    DepthSensorThor
):
    def frame_from_env(self, env: StretchManipulaTHOREnvironment, task: Optional[Task]) -> np.ndarray:
        depth = env.nav_depth.copy()
        return (depth)


class RealRGBSensorStretchManip(
    RGBSensorThor
):

    def frame_from_env(
            self, env: StretchManipulaTHOREnvironment, task: Task[StretchManipulaTHOREnvironment]
    ) -> np.ndarray:  # type:ignore
        rgb = env.manip_rgb.copy()
        return (rgb)

class RealDepthSensorStretchManip(
    DepthSensorThor
):
    def frame_from_env(self, env: StretchManipulaTHOREnvironment, task: Optional[Task]) -> np.ndarray:
        depth = env.manip_depth.copy()
        return (depth)

def normalize_real_intel_image(image,final_size=224,rotate_90_deg=False):
    assert image.shape[0] / INTEL_RESIZED_W == image.shape[1] / INTEL_RESIZED_H, ('Not right side')
    ratio = max(INTEL_RESIZED_W, INTEL_RESIZED_H) / final_size
    new_w, new_h = int(INTEL_RESIZED_W / ratio), int(INTEL_RESIZED_H / ratio)
    if (rotate_90_deg):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(image,(new_w, new_h))
    else:
        image = cv2.resize(image,(new_h, new_w))
    if len(image.shape) == 3:
        result = np.zeros((final_size, final_size, image.shape[2]))
    elif len(image.shape) == 2:
        result = np.zeros((final_size, final_size))

    new_w, new_h = image.shape[0], image.shape[1]
    start_w = int(final_size / 2 - new_w / 2)
    end_w = start_w + new_w
    start_h = int(final_size / 2 - new_h / 2)
    end_h = start_h + new_h
    result[start_w:end_w,start_h:end_h] = image
    if len(image.shape) == 2: #it is depth image
        result[result > MAX_INTEL_DEPTH] = MAX_INTEL_DEPTH
        result[result < MIN_INTEL_DEPTH] = 0
    return result.astype(image.dtype)


class StretchDetectronObjectMask(Sensor):
    def __init__(self, type: str,noise, source_camera, uuid: str = "object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.cache = {'object_name': '', 'image':None, 'mask':None}
        super().__init__(**prepare_locals_for_super(locals()))
        self.source_camera = source_camera

    @lazy_property
    def predictor(self):
        self.cfg = get_cfg()
        if platform.system() == "Darwin":
            self.cfg.MODEL.DEVICE = "cpu"
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # is this the model we want? Yes it is a pretty good model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set threshold for this model #TODO good number?
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # self.predictor = DefaultPredictor(self.cfg)
        self.class_labels = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
        return DefaultPredictor(self.cfg)


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
            #TODO NOW remove
            mask = torch.zeros((224,224,1))
            return mask

        original_im = self.source_camera.get_observation(env, task, *args, **kwargs)

        if info_to_search == self.cache['object_name']:
            if np.all(self.cache['image'] == original_im):
                return self.cache['mask']
        else:
            self.cache = {}


        # TODO VERYYYYYY IMPORTANT
        im = original_im[:,:,::-1]
        # the detection requires BGR???

        outputs = self.predictor(im * 255.)

        #  remove
        # outputs = self.predictor(im * 255.)
        # cv2.imwrite('my_image.png',im * 255.)

        #
        def visualize_detections(im, outputs ):
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            dir = 'experiment_output/visualization_predictions'
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f.png")
            os.makedirs(dir, exist_ok=True)
            print('saving all detections in ', os.path.join(dir, timestamp))
            plt.imsave(os.path.join(dir, timestamp), out.get_image()[:, :, ::-1])



        category = task.task_info[info_to_search].split('|')[0]
        assert category in THOR2COCO
        class_ind_to_look_for = self.class_labels.index(THOR2COCO[category])
        all_predicted_labels = outputs['instances'].pred_classes
        all_predicted_bbox = outputs['instances'].pred_boxes #TODO switch to segmentation



        mask = torch.zeros((im.shape[0], im.shape[1]))
        valid_boxes = [all_predicted_bbox[i] for i in range(len(all_predicted_labels)) if all_predicted_labels[i] == class_ind_to_look_for]

        for box in valid_boxes:
            x1, y1, x2, y2 = [int(x) for x in box.tensor.squeeze()]
            mask[y1:y2, x1:x2] = 1
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(224, 224)).squeeze(0).squeeze(0)

        #TODO save all the masks so we don't pass the image through detectyion multiple timesact

        # print('camera', self.source_camera, 'type', self.type, 'category', category)
        # print('observed objects', [self.class_labels[x] for x in all_predicted_labels])

        #TODO this can be optimized because we are passing this image twice and no point

        # use these later for visualization purposes?

        # plt.imsave('something.png', im)
        mask = mask.unsqueeze(-1)
        self.cache = {'object_name': info_to_search, 'image':original_im, 'mask':mask}

        return mask


class StretchObjectMask(Sensor):
    def __init__(self, type: str,noise,  uuid: str = "object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.cache = None
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if env.last_image_changed is False and self.cache is not None:
            return self.cache

        # TODO remove
        # mask = np.zeros((224, 224, 1))
        return mask


        img = env.controller.last_event.frame
        resized_image = cv2.resize(img, dsize=(224,224))
        global input_received, center_x, center_y
        center_x, center_y = -1, -1
        input_received = False

        def normalize_number(x, w):
            if x < 0:
                x = 0
            if x > w:
                x = w
            return x

        def clear_plot():
            fig.clear(); fig.clf(); plt.clf(); ax.cla(); fig.clear(True)
        def close_plot():
            clear_plot()
            plt.close(fig)
            plt.close('all')
        def onclick(event):
            global center_x, center_y, input_received
            if event.button == 3: #Right click
                center_x = -1
                center_y = -1
                input_received = True
                close_plot()
            if event.button == 1: #Left click
                center_x = event.xdata
                center_y = event.ydata
                input_received = True
                close_plot()

        import matplotlib
        matplotlib.use('MacOSX')
        fig, ax = plt.subplots()
        # clear_plot()
        ax.imshow(resized_image)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        close_plot()


        self.window_size = 20 #TODO do I want to change the size of this one maybe?
        mask = np.zeros((224, 224, 1))
        if center_y == -1 and center_x == -1:
            mask[:,:] = 0.
        else:
            offset = self.window_size / 2
            object_boundaries = center_x - offset, center_y - offset, center_x + offset, center_y + offset
            x1, y1, x2, y2 = [int(normalize_number(i, 224)) for i in object_boundaries]
            mask[y1:y2, x1:x2] = 1.
        self.cache = mask
        return mask



class RealStretchPickedUpObjSensor(Sensor):
    def __init__(self, uuid: str = "pickedup_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return False
