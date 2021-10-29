import glob
import os
import pdb

import torch
import torchvision
from PIL import Image
from torchvision import transforms
import pickle


def load_and_resize_image(imgname):
    this_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with open(imgname, 'rb') as fp:
        image = Image.open(fp).convert('RGB')
    return this_transform(image)

resnet = torchvision.models.resnet18(pretrained=True)
del resnet.fc
resnet.eval()

def calc_resnet_features(x):
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)
    x = resnet.avgpool(x)

    return x

IMAGE_DIR = 'datasets/apnd-dataset/query_images/'
feature_dir = IMAGE_DIR.replace('query_images', 'query_features')
os.makedirs(feature_dir, exist_ok=True)
all_possible_images = [f for f in glob.glob(os.path.join(IMAGE_DIR, '*', '*.png'))]

for img in all_possible_images:
    image = load_and_resize_image(img)
    with torch.no_grad():
        feature = calc_resnet_features(image.unsqueeze(0)).squeeze()

    feature_adr = img.replace(IMAGE_DIR, feature_dir).replace('.png', '.pkl')
    os.makedirs('/'.join(feature_adr.split('/')[:-1]), exist_ok=True)
    with open(feature_adr, 'wb') as f:
        pickle.dump(feature.numpy(), f)
        print('saved', feature_adr)


pdb.set_trace()
