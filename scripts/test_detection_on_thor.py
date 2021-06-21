import glob
import re

import ai2thor.controller
import torch
import cv2
import pdb

from PIL import ImageDraw

from scripts.thor_category_names import lvis_to_thor_translator

assert torch.__version__.startswith("1.8")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import numpy as np

from scripts.lvis_category_name import lvis_category_id2name, lvis_valid_categories


def convert_image(img_adr, thr=0.5, model_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", viz_only_thor_categories=True, viz_specific_category=None):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr#0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.DEVICE = "cpu"


    predictor = DefaultPredictor(cfg)



    im = cv2.imread(img_adr)
    im = cv2.resize(im, (500, 500))
    outputs = predictor(im)


    classes = (outputs["instances"].pred_classes)
    boxes = (outputs["instances"].pred_boxes)
    scores = (outputs["instances"].scores)

    out_img_adr = img_adr+'_{}_{}.png'.format(thr, model_name.split('/')[0])

    if viz_only_thor_categories:
        for i in range(len(classes)):
            c = classes[i].item()
            box = boxes[i].tensor.view(4)
            score = round(scores[i].item() * 100)
            # class_name = lvis_category_id2name[c]

            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[c]

            if class_name in lvis_to_thor_translator:
                print(class_name, 'found')

                if viz_specific_category is not None:
                    if viz_specific_category != class_name:
                        continue
                im = draw_bbox_on_img(im, lvis_to_thor_translator[class_name] + str(score), box)

        cv2.imwrite(out_img_adr, im)
    else:
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(out_img_adr, out.get_image()[:, :, ::-1])

        # pdb.set_trace()


def draw_bbox_on_img(im, class_name, box):
    x1, y1, x2, y2 = box
    # draw = ImageDraw.Draw(im)
    # draw.rectangle(((x1, y1), (x2, y2)), fill="black")
    # draw.text((y1, x1), class_name)
    color = (random.random() * 255,random.random() * 255,random.random() * 255)
    image = cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
    cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return image

# img_adr = '/Users/kianae/Desktop/sample.png'
# category = None


img_directory = '/Users/kianae/git/manipulathor/experiment_output/visualizations/PredictedBBoxSimpleDiverseBringObject_06_21_2021_13_14_03_608807/BringObjImageVisualizer_06_21_2021_13_14_03_608836'
img_adr = '/Users/kianae/git/manipulathor/experiment_output/visualizations/PredictedBBoxSimpleDiverseBringObject_06_21_2021_13_14_03_608807/BringObjImageVisualizer_06_21_2021_13_14_03_608836/06_21_2021_13_14_10_410176log_ind_0_obj_Apple_pickup_goal.png'
img_adr = '/Users/kianae/git/manipulathor/experiment_output/visualizations/PredictedBBoxSimpleDiverseBringObject_06_21_2021_13_14_03_608807/BringObjImageVisualizer_06_21_2021_13_14_03_608836/06_21_2021_13_14_10_410176log_ind_0_obj_Pan_pickup_start.png'

for img_adr in glob.glob(img_directory + '/*.png'):
    category = img_adr.split('obj_')[-1].split('_')[0]
    # category = re.sub(r'(?<!^)(?=[A-Z])', '_', category).lower().replace(' ', '_').replace('__', '_')

    convert_image(img_adr, thr=0.1)#, viz_specific_category=category)
    # convert_image_batches(img_adr, thr=0.3)
    # convert_image_batches(img_adr, thr=0.5)

    # convert_image(img_adr, thr=0.1, model_name="LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    # convert_image(img_adr, thr=0.3, model_name="LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    # convert_image(img_adr, thr=0.5, model_name="LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

    # convert_image(img_adr, thr=0.5, model_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # convert_image(img_adr, thr=0.1, model_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")


pdb.set_trace()