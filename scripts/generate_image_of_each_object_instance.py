import pdb

import ai2thor.controller
import os

import cv2
import torch
from torchvision.transforms import transforms

controller = ai2thor.controller.Controller(renderInstanceSegmentation=True, width=500, height=500)
path_to_save = '/Users/kianae/Desktop/instance_images'
os.makedirs(path_to_save, exist_ok=True)
TRAIN_OBJECTS = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug"]
TEST_OBJECTS = ["Pan", "Egg", "Spatula", "Cup"] #, 'Potato']
OBJECT_TYPES = TRAIN_OBJECTS + TEST_OBJECTS
SCENE_NAMES = ['FloorPlan{}'.format(i + 1) for i in range(30)]

def main():
    for scene in SCENE_NAMES:
        controller.reset(scene)
        for obj_type in OBJECT_TYPES:
            object_id = controller.last_event.objects_by_type(obj_type)[0]['objectId']
            controller.step(
                action="PickupObject",
                objectId=object_id,
                forceAction=True,
                manualInteract=False
            )
            frame = controller.last_event.frame.copy()

            assert object_id in controller.last_event.instance_detections2D
            obj_bbox = controller.last_event.instance_detections2D[object_id]
            x1, y1, x2, y2 = [int(k) for k in obj_bbox]
            frame = torch.Tensor(frame).permute(2, 0, 1)
            target_cropped_image = loose_crop(frame, x1, x2, y1, y2)
            transform = transforms.Resize((224, 224))

            target_cropped_image = transform(torch.Tensor(target_cropped_image))
            img_adr = os.path.join(path_to_save, f'{scene}_{obj_type}.png')
            cv2.imwrite(img_adr, (target_cropped_image.permute(1,2,0)[:,:,[2,1,0]]).int().numpy())
            controller.step(
                action="DropHandObject",
                forceAction=True
            )


def loose_crop(target_obj_image, x1, x2, y1, y2, offset=10):
    c, h, w = target_obj_image.shape
    x1 = max(x1 - offset, 0)
    y1 = max(y1 - offset, 0)
    x2 = min(x2 + offset, w)
    y2 = min(y2 + offset, h)
    crop = target_obj_image[:, y1:y2,x1:x2]
    return crop

if __name__ == '__main__':
    main()