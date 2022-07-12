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


class PoseEstimationImage(nn.Module):
    def __init__(self,
                 output_channels: int = 4):
        super().__init__()
        input_channels = 6
        #model = resnet18(pretrained=True)
        #layers = list(model.children())[:-1]
        #layers[0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.backbone = nn.Sequential(*layers)

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
        self.backbone = make_cnn(**network_args)
        self.linear = nn.Sequential(nn.Linear(512+4, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, output_channels))

    def forward(self, observations, timestep, odom):
        images = torch.cat([observations['rgb_lowres'][timestep],
                            observations['rgb_lowres_prev_frame'][timestep]], dim=-1)
        images = images.reshape(1, *images.shape)
        #images = images.permute(0, 1, 4, 2, 3)
        #images = images.reshape(-1, *images.shape[2:5])
        #features = self.backbone(images)
        features = compute_cnn_output(self.backbone, images)
        features = features.reshape(features.shape[1], features.shape[2])
        #features = features.reshape(*features.shape[:2])
        features = torch.cat([features, odom.to(torch.float)], dim=1)
        out = self.linear(features)
        out[:, 1] = 0.0
        out = out.unsqueeze(1)
        return out

class OfflineVisualOdometryDataset(Dataset):
    def __init__(self, path):
        with open(join(path, "pairs.json"), 'r') as f:
            self.pairs = json.load(f)
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



def train():
    batch_size = 32
    num_epochs = 100
    lr = 10**-3
    save_freq = 1000

    dataset = OfflineVisualOdometryDataset("/Users/karls/odom_dataset")

    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

    model = PoseEstimationImage().to(device)

    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    step = 0
    for epoch in range(num_epochs):
        data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
        for batch in tqdm(data_loader):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % save_freq == 0:
                print("losses", metrics)
                # torch.save(model.state_dict(), join(out_path, 'weights_step_{:02d}.pth'.format(step)))
        print("metrics", metrics)


if __name__ == '__main__':
    train()