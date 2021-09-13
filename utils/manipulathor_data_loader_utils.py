import random

from PIL import Image
from torchvision import transforms


def load_and_resize_image(img_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    with open(img_name, 'rb') as fp:
        image = Image.open(fp).convert('RGB')
    return transform(image)
def get_random_query_image(scene_name, object_id, query_image_dict):
    object_category = object_id.split('|')[0]
    # object_type = object_category[0].lower() + object_category[1:]
    object_type = object_category
    chosen_image_adr = random.choice(query_image_dict[object_type])
    image = load_and_resize_image(chosen_image_adr)
    return image