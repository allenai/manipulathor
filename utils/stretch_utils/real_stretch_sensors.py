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
from allenact.base_abstractions.sensor import DepthSensor, Sensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from ithor_arm.ithor_arm_sensors import DepthSensorThor
from manipulathor_utils.debugger_util import ForkedPdb
#
from utils.detection_translator_util import THOR2COCO
from utils.stretch_utils.stretch_constants import INTEL_RESIZED_H, INTEL_RESIZED_W, KINECT_REAL_W, KINECT_REAL_H, \
    MAX_INTEL_DEPTH, MIN_INTEL_DEPTH, MAX_KINECT_DEPTH, MIN_KINECT_DEPTH


class RealRGBSensorStretchIntel(
    RGBSensorThor
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(
            self, env: IThorEnvironment, task: Task[IThorEnvironment]
    ) -> np.ndarray:  # type:ignore
        rgb = env.intel_frame.copy()
        return (rgb)

class RealDepthSensorStretchIntel(
    DepthSensorThor
):
    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:
        depth = env.intel_depth.copy()
        return (depth)


class RealRGBSensorStretchKinect(
    RGBSensorThor
):

    def frame_from_env(
            self, env: IThorEnvironment, task: Task[IThorEnvironment]
    ) -> np.ndarray:  # type:ignore
        rgb = env.kinect_frame.copy()
        return (rgb)

class RealDepthSensorStretchKinect(
    DepthSensorThor
):
    def frame_from_env(self, env: IThorEnvironment, task: Optional[Task]) -> np.ndarray:
        depth = env.kinect_depth.copy()
        return (depth)

def normalize_real_intel_image(image,final_size=224):
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    if len(image.shape) == 3:
        image = image[:,:,::-1]
        #TODO why only rgb is flipped?
        image = cv2.flip(image,1)
    assert image.shape[0] / INTEL_RESIZED_W == image.shape[1] / INTEL_RESIZED_H, ForkedPdb().set_trace()
    ratio = max(INTEL_RESIZED_W, INTEL_RESIZED_H) / final_size
    new_w, new_h = int(INTEL_RESIZED_W / ratio), int(INTEL_RESIZED_H / ratio)
    image = cv2.resize(image,(new_h, new_w))
    if len(image.shape) == 3:
        result = np.zeros((final_size, final_size, image.shape[2]))
    elif len(image.shape) == 2:
        result = np.zeros((final_size, final_size))
    start_w = int(final_size / 2 - new_w / 2)
    end_w = start_w + new_w
    start_h = int(final_size / 2 - new_h / 2)
    end_h = start_h + new_h
    result[start_w:end_w,start_h:end_h] = image
    if len(image.shape) == 2: #it is depth image
        result[result > MAX_INTEL_DEPTH] = 0
        result[result < MIN_INTEL_DEPTH] = 0
    return result.astype(image.dtype)

def normalize_real_kinect_image(frame,size=224):
    assert (frame.shape[0], frame.shape[1]) == (KINECT_REAL_W, KINECT_REAL_H)
    current_size = frame.shape
    bigger_size = max(current_size[0], current_size[1])
    ratio = size / bigger_size
    w,h = (int(current_size[0] * ratio), int(current_size[1] * ratio))

    frame = cv2.resize(frame,(h,w))
    if len(frame.shape) == 3:
        result = np.zeros((size, size, frame.shape[2]))
    elif len(frame.shape) == 2:
        result = np.zeros((size, size))
    start_w = int(size / 2 - w / 2)
    end_w = start_w + w
    start_h = int(size / 2 - h / 2)
    end_h = start_h + h
    result[start_w:end_w,start_h:end_h] = frame
    if len(frame.shape) == 2: #it is depth image
        result[result > MAX_KINECT_DEPTH] = 0
        result[result < MIN_KINECT_DEPTH] = 0
    return result.astype(frame.dtype)

class StretchDetectronObjectMask(Sensor):
    def __init__(self, type: str,noise, source_camera, uuid: str = "object_mask", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        self.noise = noise
        self.cache = None
        super().__init__(**prepare_locals_for_super(locals()))
        self.source_camera = source_camera

        self.cfg = get_cfg()
        if platform.system() == "Darwin":
            self.cfg.MODEL.DEVICE = "cpu"
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # is this the model we want? Yes it is a pretty good model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model #TODO good number?
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        self.class_labels = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        if env.last_image_changed is False and self.cache is not None:
            return self.cache
        im = self.source_camera.get_observation(env, task, *args, **kwargs)
        # TODO VERYYYYYY IMPORTANT
        im = im[:,:,::-1]
        # the detection requires BGR???

        outputs = self.predictor(im)
        #
        def visualize_detections(im, outputs ):
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            dir = 'experiment_output/visualization_predictions'
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f.png")
            os.makedirs(dir, exist_ok=True)
            print('saving all detections in ', os.path.join(dir, timestamp))
            plt.imsave(os.path.join(dir, timestamp), out.get_image()[:, :, ::-1])
        # ForkedPdb().set_trace()

        if self.type == 'source':
            info_to_search = 'source_object_id'
        elif self.type == 'destination':
            info_to_search = 'goal_object_id'
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
        # use these later for visualization purposes?

        # plt.imsave('something.png', im)

        return mask.unsqueeze(-1)


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

        # ForkedPdb().set_trace()

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


#TODO these need to be redone
class RealArmPointNavEmulSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor, depth_sensor:Sensor, uuid: str = "arm_point_nav_emul", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        self.depth_sensor = depth_sensor
        uuid = '{}_{}'.format(uuid, type)

        self.min_xyz = np.zeros((3))

        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")
        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: IThorEnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return torch.zeros((3))


class RealAgentBodyPointNavEmulSensor(Sensor):

    def __init__(self, type: str, mask_sensor:Sensor, depth_sensor:Sensor, uuid: str = "point_nav_emul", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        self.mask_sensor = mask_sensor
        self.depth_sensor = depth_sensor
        uuid = '{}_{}'.format(uuid, type)

        self.min_xyz = np.zeros((3))
        self.dummy_answer = torch.zeros(3)
        self.dummy_answer[:] = 4 # is this good enough?
        self.device = torch.device("cpu")


        super().__init__(**prepare_locals_for_super(locals()))
    def get_observation(
            self, env: IThorEnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return torch.zeros((3))
#TODO are we sure that we don't need to do anything about the depth normalization? we have to clip depth from intel realsense so that the norms are similar to the depth we get from thor? do we do the postprocessing for depth iamges?



#With open CV
# def get_observation(
#         self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
# ) -> Any:
#     if env.last_image_changed is False and self.cache is not None:
#         return self.cache
#
#     # mask = np.zeros((224, 224, 1))
#     # mask[90:110, 90:110] = 1
#     # mask[90:110, :20] = 1
#     img = env.current_frame
#     resized_image = cv2.resize(img, dsize=(224,224))
#     global input_received, center_x, center_y
#     center_x, center_y = -1, -1
#     input_received = False
#
#     def on_click(event, x, y, p1, p2):
#         global center_x, center_y, input_received
#         if event == cv2.EVENT_LBUTTONDOWN:
#             center_x = x
#             center_y = y
#             print((x, y))
#             input_received = True
#             cv2.destroyWindow("image")
#             cv2.destroyAllWindows()
#
#         if event == cv2.EVENT_RBUTTONDOWN:
#             center_x = -1
#             center_y = -1
#             print((-1,-1))
#             input_received = True
#
#     def normalize_number(x, w):
#         if x < 0:
#             x = 0
#         if x > w:
#             x = w
#         return x
#
#     cv2.imshow("image", resized_image[:,:,[2,1,0]])
#     cv2.setMouseCallback('image', on_click)
#     while not input_received:
#         k = cv2.waitKey(100)
#         # if k == 27:
#         #     print('ESC')
#         #     cv2.destroyAllWindows()
#         #     break
#         # if cv2.getWindowProperty('image',1) == -1 :
#         #     break
#     cv2.destroyWindow("image")
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)
#
#     self.window_size = 20 TODO do I want to change the size of this one maybe?
#     mask = np.zeros((224, 224, 1))
#     if center_y == -1 and center_x == -1:
#         mask[:,:] = 0.
#     else:
#         offset = self.window_size / 2
#         object_boundaries = center_x - offset, center_y - offset, center_x + offset, center_y + offset
#         x1, y1, x2, y2 = [int(normalize_number(i, 224)) for i in object_boundaries]
#         mask[y1:y2, x1:x2] = 1.
#     self.cache = mask
#     return mask