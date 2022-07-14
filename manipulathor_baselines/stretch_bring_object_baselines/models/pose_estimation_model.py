import torch
import torch.nn as nn
from torch.utils.data import Dataset

from os.path import join
import json

from allenact.utils.model_utils import make_cnn, compute_cnn_output

import numpy as np
import imageio

from manipulathor_baselines.stretch_bring_object_baselines.models.pose_estimation_loss import calc_pose_estimation_loss

from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
import os

class PoseEstimationImage(nn.Module):
    def __init__(self,
                 resnet=True,
                 output_channels: int = 4):
        super().__init__()
        input_channels = 8
        self.resnet = resnet
        self.backbone = self.make_backbone(input_channels)

        #self.backbone_arm = self.make_backbone(input_channels)

        self.linear = nn.Sequential(nn.Linear(512+512+4, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, output_channels))

    def make_backbone(self, input_channels):
        if self.resnet:
            model = resnet18(pretrained=True)
            layers = list(model.children())[:-1]
            layers[0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            backbone = nn.Sequential(*layers)
        else:
            num_features = 512
            network_args = {'input_channels': input_channels,
                            'layer_channels': [32, 64, 32],
                            'kernel_sizes': [(8, 8), (4, 4), (3, 3)],
                            'strides': [(4, 4), (2, 2), (1, 1)],
                            'paddings': [(0, 0), (0, 0), (0, 0)],
                            'dilations': [(1, 1), (1, 1), (1, 1)],
                            'output_height': 24,
                            'output_width': 24,
                            'output_channels': num_features,
                            'flatten': True,
                            'output_relu': True}
            backbone = make_cnn(**network_args)
        return backbone

    def encode_images(self, images, backbone):
        images = images.reshape(1, *images.shape)
        if self.resnet:
            images = images.permute(0, 1, 4, 2, 3)
            images = images.reshape(-1, *images.shape[2:5])
            features = backbone(images)
            features = features.reshape(*features.shape[:2])
        else:
            features = compute_cnn_output(backbone, images)
            features = features.reshape(features.shape[1], features.shape[2])
        return features

    def forward(self, observations, timestep, odom):
        images = torch.cat([observations['rgb_lowres'][timestep],
                            observations['depth_lowres'][timestep],
                            observations['rgb_lowres_prev_frame'][timestep],
                            observations['depth_lowres_prev_frame'][timestep]], dim=-1)
        
        images_arm = torch.cat([observations['rgb_arm_lowres'][timestep],
                            observations['depth_arm_lowres'][timestep],
                            observations['rgb_arm_lowres_prev_frame'][timestep],
                            observations['depth_arm_lowres_prev_frame'][timestep]], dim=-1)

        features = self.encode_images(images, self.backbone)
        features_arm = self.encode_images(images_arm, self.backbone)
        features = torch.cat([features, features_arm, odom.to(torch.float)], dim=1)
        #images_combined = torch.cat([images, images_arm], dim=0)
        #features = self.encode_images(images_combined, self.backbone)
        
        #features = torch.cat([features[:images.shape[0]], features[images.shape[0]:], odom.to(torch.float)], dim=1)
        out = self.linear(features)
        out[:, 1] = 0.0
        out = out.unsqueeze(1)
        return out

class OfflineVisualOdometryDataset(Dataset):
    def __init__(self, path):
        self.pairs = []

        pairs_files = [f for f in os.listdir(path) if os.path.isfile(join(path, f)) and 'pairs' in f]
        for pair in pairs_files:
            with open(join(path, "pairs.json"), 'r') as f:
                self.pairs.extend(json.load(f))
        self.path = path

        self.max_depth = 5.0

    
    def __len__(self):
        return len(self.pairs)
    
    def convert_uint16_depth_to_metric(self, depth):
        depth = depth / 65536.0 * self.max_depth
        depth = np.asarray(depth)
        depth = depth.astype(np.float32)
        depth = depth.reshape(*depth.shape, 1)
        return depth
    
    def normalize_image(self, image):
        image = image / 255.0
        image = image - np.array([[0.485, 0.456, 0.406]])
        image = image / np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float32)
        # image = image.transpose(2, 0, 1)
        image = np.asarray(image)
        return image

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        odom = pair['noisy_translation'].copy()
        odom.append(pair['noisy_rotation'])
        odom = np.array(odom)

        target = pair['gt_translation'].copy()
        target.append(pair['gt_rotation'])
        target = np.array(target).reshape(1, 4)

        data = {'odom': odom,
                'target': target}

        episode = pair['episode']
        for index, name in ((pair['step'], ''), (pair['step'] - 1, '_prev_frame')):
            im = imageio.imread(join(self.path, 'rgb_episode{:05d}_step{:05d}.jpg'.format(episode, index)))
            data['rgb_lowres'+name] = self.normalize_image(im)
            im_arm = imageio.imread(join(self.path,'rgb_arm_episode{:05d}_step{:05d}.jpg'.format(episode, index)))
            data['rgb_arm_lowres'+name] = self.normalize_image(im_arm)


            depth = imageio.imread(join(self.path, 'depth_episode{:05d}_step{:05d}.png'.format(episode, index)))
            data['depth_lowres'+name] = self.convert_uint16_depth_to_metric(depth)
            depth_arm = imageio.imread(join(self.path, 'depth_arm_episode{:05d}_step{:05d}.png'.format(episode, index)))
            data['depth_arm_lowres'+name] = self.convert_uint16_depth_to_metric(depth_arm)

        
        return data


def eval_model(model, batch, device):

    for k in batch.keys():
        batch[k] = batch[k].to(device)
        if 'depth' in k or 'rgb' in k:
            batch[k] = batch[k].view(1, *batch[k].shape)

    out = model(batch, 0, batch['odom'])

    pose_errors = out - batch['target']
    position_errors = pose_errors[:, :, :3]
    rotation_errors = pose_errors[:, :, -1]

    odom_errors = batch['odom'] - batch['target']
    odom_pos_errors = odom_errors[:, :, :3]
    odom_rot_errors = odom_errors[:, :, -1]
    loss, metrics = calc_pose_estimation_loss(position_errors, rotation_errors, odom_pos_errors, odom_rot_errors)
    return loss, metrics

def train():
    batch_size = 128
    num_epochs = 100
    lr = 10**-3
    save_freq = 1000
    tb_freq = 10
    out_path = "experiment_output/tb/pose_estimation"
    name = "test_resnet_depth_arm"
    out_path = join(out_path, name)


    device = 'cpu' if not torch.cuda.is_available() else 'cuda:1'
    path = "/Users/karls/odom_dataset" if device == 'cpu' else "/home/karls/odom_dataset"
    valid_path = "/Users/karls/odom_dataset_valid" if device == 'cpu' else "/home/karls/odom_dataset_valid"
    num_workers = 0 if device == 'cpu' else 4

    dataset = OfflineVisualOdometryDataset(path)
    valid_dataset = OfflineVisualOdometryDataset(valid_path)
    model = PoseEstimationImage().to(device)

    optimizer = torch.optim.Adam(model.parameters())

    os.makedirs(out_path)

    writer = SummaryWriter(out_path)

    model.train()
    step = 0
    for epoch in range(num_epochs):
        data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
        for batch in tqdm(data_loader):
            loss, metrics = eval_model(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % tb_freq == 0:
                for k in metrics:
                    writer.add_scalar("train-losses/pose_loss/"+k, metrics[k], step)

            if step % save_freq == 0:
                print("losses", metrics)
                model.eval()
                valid_data_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=num_workers)
                all_metrics = {}
                num_samples = 0
                for batch in tqdm(valid_data_loader):
                    with torch.no_grad():
                        loss, metrics = eval_model(model, batch, device)
                        batch_size = batch['target'].shape[0]
                        num_samples += batch_size
                        for k in metrics:
                            if k not in all_metrics:
                                all_metrics[k] = metrics[k] * batch_size
                            else:
                                all_metrics[k] += metrics[k] * batch_size
                for k in all_metrics:
                    all_metrics[k] /= num_samples
                    writer.add_scalar("valid-metrics/pose_loss/"+k, all_metrics[k], step)
                print("valid", all_metrics)

                model.train()
                torch.save(model.state_dict(), join(out_path, 'weights_step_{:02d}.pth'.format(step)))
    torch.save(model.state_dict(), join(out_path, 'weights_step_{:02d}.pth'.format(step)))


if __name__ == '__main__':
    train()
