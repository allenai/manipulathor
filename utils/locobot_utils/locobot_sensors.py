"""Utility classes and functions for sensory inputs used by the models."""
import glob
import os
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

from ithor_arm.arm_calculation_utils import (
    convert_world_to_agent_coordinate,
    convert_state_to_tensor,
    diff_position,
)
from ithor_arm.ithor_arm_environment import ManipulaTHOREnvironment
from manipulathor_utils.debugger_util import ForkedPdb
#
# class LocoBotRawRGBSensorThor(Sensor):
#     def __init__(self, uuid: str = "raw_rgb_lowres", **kwargs: Any):
#         observation_space = gym.spaces.Box(
#             low=0, high=1, shape=(1,), dtype=np.float32
#         )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
#         super().__init__(**prepare_locals_for_super(locals()))
#
#     def get_observation(
#             self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
#     ) -> Any:
#         # env.controller.step('Initialize') #TODO seriously?
#         #TODO this was env.current_frame. maybe that is not updated?
#         #TODO whatever we do here we should do for depth
#
#         # frame = env.controller.last_event.frame.copy()
#         # frame = cv2.resize(frame, dsize=(224,224))
#         # return frame
#
#         return env.current_frame.copy()


class LocoBotCategorySampleSensor(Sensor):
    def __init__(self, type: str, uuid: str = "category_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.type = type
        uuid = '{}_{}'.format(uuid, type)
        super().__init__(**prepare_locals_for_super(locals()))


    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:

        if self.type == 'source':
            info_to_search = 'source_object_query'
        elif self.type == 'destination':
            info_to_search = 'goal_object_query'
        else:
            raise Exception('Not implemented', self.type)
        image = task.task_info[info_to_search]
        return image


class LocoBotObjectMask(Sensor):
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

        #TODO there seems to be a lag is this fixed?
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



class LocoBotPickedUpObjSensor(Sensor):
    def __init__(self, uuid: str = "pickedup_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return False



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
#     self.window_size = 20 #TODO do I want to change the size of this one maybe?
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