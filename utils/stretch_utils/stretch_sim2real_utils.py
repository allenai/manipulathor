import random

from manipulathor_utils.debugger_util import ForkedPdb
from utils.stretch_utils.stretch_constants import INTEL_RESIZED_W, INTEL_RESIZED_H, KINECT_RESIZED_W, KINECT_RESIZED_H, \
    MAX_INTEL_DEPTH, MIN_INTEL_DEPTH, MAX_KINECT_DEPTH, MIN_KINECT_DEPTH
import numpy as np
import cv2

def kinect_reshape(frame):
    frame = frame.copy()

    #TODO
    return frame

    desired_w, desired_h = KINECT_RESIZED_W, KINECT_RESIZED_H
    # original_size = desired_h
    assert frame.shape[0] == frame.shape[1]
    original_size = frame.shape[0]
    fraction = max(desired_h, desired_w) / original_size
    beginning = original_size / 2 - desired_w / fraction / 2
    end = original_size / 2 + desired_w / fraction / 2
    frame[:int(beginning), :] = 0
    frame[int(end):, :] = 0
    if len(frame.shape) == 2: #it is depth image
        frame[frame > MAX_KINECT_DEPTH] = MAX_KINECT_DEPTH
        frame[frame < MIN_KINECT_DEPTH] = 0
    # if len(frame.shape) == 3 and frame.shape[2] == 3:
    #     # frame = clip_rgb_kinect_frame(frame)
    #     pass
    if (len(frame.shape) == 3 and frame.shape[2] ==1) or len(frame.shape) == 2:
        cropped_frame = frame[int(beginning):int(end), :]
        frame[int(beginning):int(end), :] = clip_depth_kinect_frame(cropped_frame)

    return frame



def intel_reshape(frame):
    frame = frame.copy()

    return frame #TODO

    desired_w, desired_h = INTEL_RESIZED_W, INTEL_RESIZED_H
    assert frame.shape[0] == frame.shape[1]
    original_size = frame.shape[0]
    fraction = max(desired_h, desired_w) / original_size
    beginning = original_size / 2 - desired_h / fraction / 2
    end = original_size / 2 + desired_h / fraction / 2
    frame[:,:int(beginning)] = 0
    frame[:,int(end):] = 0
    if len(frame.shape) == 2: #it is depth image
        frame[frame > MAX_INTEL_DEPTH] = MAX_INTEL_DEPTH
        frame[frame < MIN_INTEL_DEPTH] = 0
    return frame

DEPTH_KINECT_MASK_FRAMES = None
RGB_KINECT_MASK_FRAMES = None
remake_mask_prob = 0.1
PIX_NOISE = 20

def clip_depth_kinect_frame(frame):
    return frame #TODO
    global DEPTH_KINECT_MASK_FRAMES
    if len(frame.shape) == 2:
        w, h = frame.shape
    if len(frame.shape) == 3:
        w, h, c = frame.shape
    if DEPTH_KINECT_MASK_FRAMES is None or DEPTH_KINECT_MASK_FRAMES.shape[0] != w or DEPTH_KINECT_MASK_FRAMES.shape[1] != h or random.random() < remake_mask_prob:
        DEPTH_KINECT_MASK_FRAMES = set_mask_kinect_converted_depth(w, h)
    frame[(1 - DEPTH_KINECT_MASK_FRAMES).astype(bool)] = 0
    return frame

# def clip_rgb_kinect_frame(frame):
#     global RGB_KINECT_MASK_FRAMES
#     if len(frame.shape) == 2:
#         w, h = frame.shape
#     if len(frame.shape) == 3:
#         w, h, c = frame.shape
#     if RGB_KINECT_MASK_FRAMES is None or RGB_KINECT_MASK_FRAMES.shape[0] != w or RGB_KINECT_MASK_FRAMES.shape[1] != h or random.random() < remake_mask_prob:
#         RGB_KINECT_MASK_FRAMES = set_mask_kinect_rgb(w, h)
#     frame[(1 - RGB_KINECT_MASK_FRAMES).astype(bool)] = 0
#     return frame

def set_mask_kinect_converted_depth(w, h):

    original_w, original_h = 1280, 720
    w_left_border, w_right_border = 140, 200
    w_up_left, h_up_left = 330, 300
    w_up_right, h_up_right = 360,330
    w_down_left, h_down_left = 320,280
    w_down_right, h_down_right = 360, 280

    w_up_left, h_up_left,    w_up_right, h_up_right,    w_down_left, h_down_left,    w_down_right, h_down_right, w_left_border, w_right_border = tweak_mask_values([w_up_left, h_up_left,    w_up_right, h_up_right,    w_down_left, h_down_left,    w_down_right, h_down_right, w_left_border, w_right_border])
    # init = [(0, 0), (original_w, 0), (0, original_h), (original_w, original_h)]
    # ws = [w_up_left, -w_up_right, w_down_left, -w_down_right]
    # hs = [h_up_left, h_up_right, -h_down_left, -h_down_right]
    init = [(w_left_border, 0), (original_w - w_right_border, 0), (w_left_border, original_h), (original_w - w_right_border, original_h)]
    ws = [w_up_left, original_w-w_up_right, w_down_left, original_w-w_down_right]
    hs = [h_up_left, h_up_right, original_h-h_down_left, original_h-h_down_right]

    mask = np.ones((original_h, original_w))

    for i in range(4):
        pt1 = init[i]
        pt2 = (ws[i], pt1[1])
        pt3 = (pt1[0], hs[i])
        triangle_cnt = np.array( [pt1, pt2, pt3] )


        mask = cv2.drawContours(mask, [triangle_cnt], 0, (0), -1)

    mask[:,:w_left_border] = 0
    mask[:,-w_right_border:] = 0
    # cv2.imwrite('something.png', MASK_FRAMES * 255.)
    MASK_FRAMES = cv2.resize(mask, (h, w))
    return MASK_FRAMES

def set_mask_kinect_depth_original(w, h):
    original_w, original_h = 640, 576
    w_up_left, h_up_left = 150, 270
    w_up_right, h_up_right = 120, 210
    w_down_left, h_down_left = 150,250
    w_down_right, h_down_right = 120, 200

    w_up_left, h_up_left,    w_up_right, h_up_right,    w_down_left, h_down_left,    w_down_right, h_down_right = tweak_mask_values([w_up_left, h_up_left,    w_up_right, h_up_right,    w_down_left, h_down_left,    w_down_right, h_down_right])
    init = [(0, 0), (original_w, 0), (0, original_h), (original_w, original_h)]
    ws = [w_up_left, -w_up_right, w_down_left, -w_down_right]
    hs = [h_up_left, h_up_right, -h_down_left, -h_down_right]

    mask = np.ones((original_h, original_w))

    for i in range(4):
        pt1 = init[i]
        pt2 = (pt1[0] + ws[i], pt1[1])
        pt3 = (pt1[0], pt1[1] + hs[i])
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        mask = cv2.drawContours(mask, [triangle_cnt], 0, (0), -1)
    MASK_FRAMES = cv2.resize(mask, (h, w))
    return MASK_FRAMES

# def set_mask_kinect_rgb(w, h):
#     original_w, original_h = 640, 576
#     w_up_left, h_up_left = 150, 270
#     w_up_right, h_up_right = 120, 210
#     w_down_left, h_down_left = 150,250
#     w_down_right, h_down_right = 120, 200
#
#     w_up_left, h_up_left,    w_up_right, h_up_right,    w_down_left, h_down_left,    w_down_right, h_down_right = tweak_mask_values([w_up_left, h_up_left,    w_up_right, h_up_right,    w_down_left, h_down_left,    w_down_right, h_down_right])
#
#     init = [(0, 0), (original_w, 0), (0, original_h), (original_w, original_h)]
#     ws = [w_up_left, -w_up_right, w_down_left, -w_down_right]
#     hs = [h_up_left, h_up_right, -h_down_left, -h_down_right]
#
#     mask = np.ones((original_h, original_w))
#
#     for i in range(4):
#         pt1 = init[i]
#         pt2 = (pt1[0] + ws[i], pt1[1])
#         pt3 = (pt1[0], pt1[1] + hs[i])
#         triangle_cnt = np.array( [pt1, pt2, pt3] )
#         mask = cv2.drawContours(mask, [triangle_cnt], 0, (0), -1)
#     OFFSET_UP, OFFSET_DOWN = 45, 40
#     OFFSET_UP, OFFSET_DOWN = tweak_mask_values([OFFSET_UP, OFFSET_DOWN])
#     mask[:OFFSET_UP,:] = 0
#     mask[-OFFSET_DOWN:,:] = 0
#     MASK_FRAMES = cv2.resize(mask, (h, w))
#     return MASK_FRAMES

def tweak_mask_values(list_to_be_tweaked, value_to_change=PIX_NOISE):
    return [x + random.randint(-value_to_change,value_to_change) for x in list_to_be_tweaked]
