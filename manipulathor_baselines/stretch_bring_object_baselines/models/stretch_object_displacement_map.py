""" Model for object displacement

By Karl Schmeckpeper
"""

from datetime import datetime
from typing import Tuple, Optional, Dict

import torch
from torch import nn
import torch.nn.functional as F

import cv2

import gym
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import depth_frame_to_world_space_xyz, \
    project_point_cloud_to_map, depth_frame_to_camera_space_xyz, camera_space_xyz_to_world_xyz
from allenact.embodiedai.mapping.mapping_utils.map_builders import BinnedPointCloudMapBuilder

from utils.model_utils import LinearActorHeadNoCategory

from torchvision.models import resnet18

class PoseEstimator(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int = 4,
                 num_features: int = 512):
        super().__init__()
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
        self.linear = nn.Linear(num_features, output_channels)
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self,
                x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.reshape(1, *x.shape)
        features = compute_cnn_output(self.backbone, x)
        out = self.linear(features)

        # Hard codes elevation change to be zero
        out[:, :, 1] = 0.0
        return out


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


class StretchObjectDisplacementMapModel(ActorCriticModel[CategoricalDistr]):
    """

    """

    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            hidden_size=512,
            map_size=224,
            trainable_masked_hidden_state: bool = False,
            num_rnn_layers=1,
            rnn_type="GRU",
            learn_pose=True,
            visualize=False,
            ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._learn_pose = learn_pose

        self._hidden_size = hidden_size
        self._visualize = visualize

        self.map_channels = 4
        self.map_size = map_size
        self.map_resolution_cm = 5
        self.min_xyz = torch.tensor([map_size / 2. * self.map_resolution_cm / 100,
                                     0.0,
                                     map_size / 2. * self.map_resolution_cm / 100])

        network_args = {'input_channels': 5, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)
        network_args = {'input_channels': 5, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder_arm = make_cnn(**network_args)

        if self._learn_pose:
            #self.pose_estimation = PoseEstimator(input_channels=2*self.map_channels,
            #                                     output_channels=4)
            self.pose_estimation = PoseEstimationImage()

        network_args = {'input_channels': self.map_channels, 'layer_channels': [32, 64, 32], 'kernel_sizes': [(8, 8), (4, 4), (3, 3)], 'strides': [(4, 4), (2, 2), (1, 1)], 'paddings': [(0, 0), (0, 0), (0, 0)], 'dilations': [(1, 1), (1, 1), (1, 1)], 'output_height': 24, 'output_width': 24, 'output_channels': 512, 'flatten': True, 'output_relu': True}
        self.full_visual_encoder_map = make_cnn(**network_args)

        self.body_pointnav_embedding = nn.Sequential(
             nn.Linear(3, 32),
             nn.LeakyReLU(),
             nn.Linear(32, 128),
             nn.LeakyReLU(),
             nn.Linear(128, 512),
        )
        self.arm_pointnav_embedding = nn.Sequential(
             nn.Linear(3, 32),
             nn.LeakyReLU(),
             nn.Linear(32, 128),
             nn.LeakyReLU(),
             nn.Linear(128, 512),
        )

        self.state_encoder = RNNStateEncoder(
            #512 * 3,
            # 512 * 4,
            512 * 5,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pickup = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_pickup = LinearCriticHead(self._hidden_size)

        
        # Bins for zero elevation being at ground
        # self.bins = [0.2, 1.5]

        # bins for zero elevation being at the robot's origin
        self.bins = [-0.7, 0.6]

        self.train()



        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))

        self.step_count = 0
        
        self.camera_poses = []
        self.agent_poses = []

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            map=(
                (
                    ("layer", 1),
                    ("sampler", None),
                    ("hidden", self.map_size * self.map_size * self.map_channels)
                ),
                torch.float32
            ),
            prev_pose=(
                (
                    ("layer", 1),
                    ("sampler", None),
                    ("hidden", 4)
                ),
                torch.float32
            )
        )


    def transform_global_map_to_ego_map(
            self,
            global_maps: torch.FloatTensor,
            agent_info: Dict,
        ) -> torch.FloatTensor:

        ego_rotations = -torch.deg2rad(agent_info['rotation'].reshape(-1))
        ego_xyz = agent_info['xyz'].reshape(-1, 3) / (self.map_resolution_cm / 100.) / self.map_size * 2.0
        transform_world_to_ego = torch.tensor([[[torch.cos(a), -torch.sin(a), pos[0]],
                                                [torch.sin(a),  torch.cos(a), pos[2]]] for a, pos in zip(ego_rotations, ego_xyz)], 
                                                dtype=global_maps.dtype, device=global_maps.device)
        global_maps_reshaped = global_maps.reshape(-1, *global_maps.shape[2:]).permute(0, 3, 1, 2)

        affine_grid_world_to_ego = F.affine_grid(transform_world_to_ego, global_maps_reshaped.shape)
        ego_maps = F.grid_sample(global_maps_reshaped, affine_grid_world_to_ego, align_corners=False)
        # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
        ego_maps = ego_maps.permute(0, 2, 3, 1).reshape(global_maps.shape)
        # print("egomaps", ego_maps.shape)
        # cv2.imwrite("/Users/karls/debug_images/act_all_map_{}_ego.png".format(self.step_count), ego_maps[-1, 0].detach().cpu().numpy()*256)
        return ego_maps

    def transform_by_relative_pos(
            self,
            prev_map: torch.FloatTensor,
            pose: torch.FloatTensor,
        ) -> torch.FloatTensor:
        relative_rot = -torch.deg2rad(pose[:, :, -1])
        relative_xyz = pose[:, :, :3].reshape(-1, 3) / (self.map_resolution_cm / 100.) / self.map_size * 2.0

        transform_prev_to_curr = torch.tensor([[[torch.cos(a), -torch.sin(a), pos[0]],
                                                [torch.sin(a),  torch.cos(a), pos[2]]] for a, pos in zip(relative_rot, relative_xyz)],
                                              dtype=prev_map.dtype, device=prev_map.device)
        
        prev_map_reshaped = prev_map.permute(0, 3, 1, 2)

        affine_grid_prev_to_curr = F.affine_grid(transform_prev_to_curr, prev_map_reshaped.shape)
        prev_map_in_curr_frame = F.grid_sample(prev_map_reshaped, affine_grid_prev_to_curr, align_corners=False)

        prev_map_in_curr_frame = prev_map_in_curr_frame.permute(0, 2, 3, 1)
        return prev_map_in_curr_frame

    def get_binned_map(
            self,
            frame: torch.FloatTensor,
            mask: torch.Tensor,
            camera: Dict, 
            timestep: int,
            pose: torch.FloatTensor,
            egocentric: bool = False,
        ) -> torch.FloatTensor:
        position = pose[:, :, :3]
        rotation = pose[:, :, -1]
        binned_updates = []
        for i in range(frame.shape[0]):
            if torch.sum(mask[i]) == 0:
                binned_updates.append(torch.zeros(*frame[i].shape, len(self.bins)+1, device=frame.device))
                continue

            camera_space_xyz = depth_frame_to_camera_space_xyz(frame[i], mask[i], fov=camera['fov'][timestep][i])
            # camera_xyz = camera['xyz'][timestep][i]
            # camera_rot = camera['rotation'][timestep][i]
            # if egocentric:
            #     camera_xyz = camera['xyz_offset'][timestep][i]
            #     camera_rot = camera['rotation_offset'][timestep][i]
            # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
            xyz_offset = camera['xyz_offset'][timestep][i].reshape(3)
            camera_xyz = position[i].clone().reshape(3)
            agent_rot = rotation[i].reshape([])

            camera_xyz[0] += xyz_offset[0] * torch.cos(torch.deg2rad(-agent_rot)) - xyz_offset[2] * torch.sin(torch.deg2rad(-agent_rot))
            camera_xyz[1] += xyz_offset[1]
            camera_xyz[2] += xyz_offset[0] * torch.sin(torch.deg2rad(-agent_rot)) + xyz_offset[2] * torch.cos(torch.deg2rad(-agent_rot))
            # camera_xyz = torch.tensor([cos_of_rot * camera_xyz[0] - sin_of_rot * camera_xyz[2],
            #                            camera_xyz[1],
            #                            sin_of_rot * camera_xyz[0] + cos_of_rot * camera_xyz[2]])
            camera_rot = agent_rot + camera['rotation_offset'][timestep][i].reshape([])

            world_points = camera_space_xyz_to_world_xyz(camera_space_xyzs=camera_space_xyz, 
                                                        camera_world_xyz=camera_xyz, 
                                                        rotation=camera_rot.cpu(), 
                                                        horizon=camera['horizon'][timestep][i].cpu())
            world_points = world_points.permute(1, 0).unsqueeze(0).to(frame.device)
            world_points_plus_min = world_points + self.min_xyz.to(frame.device)
            binned_map_update = project_point_cloud_to_map(xyz_points = world_points_plus_min,
                                                        bin_axis="y",
                                                        bins=self.bins,
                                                        map_size=self.map_size,
                                                        resolution_in_cm=self.map_resolution_cm,
                                                        flip_row_col=True)

            binned_updates.append(binned_map_update)
        binned_updates = torch.stack(binned_updates)
        return binned_updates

    def project_depth_to_map(
        self,
        observations: ObservationType,
        pose: torch.FloatTensor,
        timestep: int,
        batch_size: int,
    ) -> torch.FloatTensor:

        map_update = torch.zeros(batch_size, self.map_size, self.map_size, self.map_channels, device=pose.device)
        for camera_name, frame, source_mask, destination_mask in \
                zip(observations['odometry_emul']['camera_info'],
                    (observations['depth_lowres'], observations['depth_lowres_arm']),
                    (observations['object_mask_source'], observations['object_mask_kinect_source']),
                    (observations['object_mask_destination'], observations['object_mask_kinect_destination'])):
            camera = observations['odometry_emul']['camera_info'][camera_name]

            reshaped_frame = frame[timestep].reshape(batch_size, *frame.shape[2:4]).clone()
            
            valid_depths = torch.logical_and(reshaped_frame>0.01, reshaped_frame < 2.99)

            # Filters out points detected on the arm
            # This is probably too conservative. We should be able to compute a more accurate mask from the arm pose
            if camera_name == 'camera_arm':
                valid_depths = torch.logical_and(reshaped_frame > 1.0, valid_depths)
            if torch.sum(valid_depths) == 0:
                continue
            
            binned_map_update = self.get_binned_map(reshaped_frame, valid_depths, camera, timestep, pose)

            # Discards ceiling detections when updating the map
            map_update[:, :, :, :2] += binned_map_update[:, :, :, :2]

            # Adds the source object detections
            source_object_mask = torch.logical_and(valid_depths, source_mask[timestep].reshape(valid_depths.shape) > 0.0)
            if torch.sum(source_object_mask) > 0:
                source_map_update = self.get_binned_map(reshaped_frame, source_object_mask, camera, timestep, pose)
                source_map_update = torch.sum(source_map_update, dim=-1)
                map_update[:, :, :, 2] += source_map_update

            # Adds the target object detections
            destination_object_mask = torch.logical_and(valid_depths, destination_mask[timestep].reshape(valid_depths.shape) > 0.0)
            if torch.sum(destination_object_mask) > 0:
                destination_map_update = self.get_binned_map(reshaped_frame, destination_object_mask, camera, timestep, pose)
                destination_map_update = torch.sum(destination_map_update, dim=-1)
                map_update[:, :, :, 3] += destination_map_update
        return map_update


    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        pose_errors = []

        batch_size = memory.tensor("map").shape[1]
        # print("batch size", batch_size)
        current_map = memory.tensor("map").reshape(batch_size, self.map_size, self.map_size, self.map_channels)
        # print("min max", torch.min(observations['depth_lowres']), torch.max(observations['depth_lowres']))
        prev_pose = memory.tensor("prev_pose").reshape(batch_size, 1, 4)
        # Builds geocentric maps
        all_maps = [current_map]
        for timestep in range(observations['depth_lowres'].shape[0]):
            all_maps.append(all_maps[-1].clone() * masks[timestep].reshape(batch_size, 1, 1, 1))
            prev_pose = prev_pose * masks[timestep].unsqueeze(-1)
            if self._learn_pose:
                # Uses noisy pose to get estimate of previous map location in current frame
                # odom_update = torch.cat([observations['odometry_emul']['agent_info']['relative_xyz'],
                #                          observations['odometry_emul']['agent_info']['relative_rot'].unsqueeze(-1)], dim=-1)
                sin_of_prev = torch.sin(torch.deg2rad(prev_pose[:, :, -1])).squeeze()
                cos_of_prev = torch.cos(torch.deg2rad(prev_pose[:, :, -1])).squeeze()
                relative_xyz = observations['odometry_emul']['agent_info']['noisy_relative_xyz'][timestep]
                odom_update_xyz = torch.zeros_like(relative_xyz)
                # if prev_pose.shape[0] > 2:
                # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
                odom_update_xyz[:, 0] = - cos_of_prev * relative_xyz[:, 0] + sin_of_prev * relative_xyz[:, 2]
                odom_update_xyz[:, 2] = sin_of_prev * relative_xyz[:, 0] + cos_of_prev * relative_xyz[:, 2]
                odom_update = torch.cat([odom_update_xyz,
                                         observations['odometry_emul']['agent_info']['noisy_relative_rot'][timestep].unsqueeze(-1)], dim=-1)

                if False:
                    #from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
                    prev_pose = prev_pose + odom_update.unsqueeze(1)
                    prev_map_in_ego_with_error = self.transform_by_relative_pos(all_maps[-1], prev_pose)

                    # generate map of current frame in egocentric view
                    egocentric_pose = torch.zeros_like(prev_pose)
                    # egocentric_pose = {'xyz': torch.zeros_like(observations['odometry_emul']['agent_info']['xyz']),
                    #                    'rotation': torch.zeros_like(observations['odometry_emul']['agent_info']['rotation'])}
                    current_map_in_ego = self.project_depth_to_map(observations, egocentric_pose, timestep, batch_size)

                    # cv2.imwrite("/Users/karls/debug_images/map_{}_{}_prev.png".format(self.step_count, timestep), prev_map_in_ego_with_error[0, :, :, :3].detach().cpu().numpy()*255)
                    # cv2.imwrite("/Users/karls/debug_images/map_{}_{}_curr.png".format(self.step_count, timestep), current_map_in_ego[0, :, :, :3].detach().cpu().numpy()*255)
                    stacked_maps = torch.cat([prev_map_in_ego_with_error, current_map_in_ego], dim=-1)

                    
                    pose_update = self.pose_estimation(stacked_maps)
                else:
                    pose_update = self.pose_estimation(observations, timestep, odom_update)
                pose = prev_pose + pose_update
                position_error = observations['odometry_emul']['agent_info']['xyz'][timestep].unsqueeze(1) - pose[:, :, :3]
                rotation_error = observations['odometry_emul']['agent_info']['rotation'][timestep].unsqueeze(1) - pose[:, :, -1]

                odom_pos_error = observations['odometry_emul']['agent_info']['xyz'][timestep].unsqueeze(1) - prev_pose[:, :, :3]
                odom_rot_error = observations['odometry_emul']['agent_info']['rotation'][timestep].unsqueeze(1) - prev_pose[:, :, -1]
#                odom_rotations = odom_rot_error.clone()
#                odom_rotations[torch.where(odom_rotations > 180)[0]] -= 360
#                odom_rotations[torch.where(odom_rotations < -180)[0]] += 360
#                if torch.max(torch.abs(odom_rotations)) > 15.0:
#                    from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
                pose_errors.append({'position': position_error, 'rotation': rotation_error, 'odom_pos': odom_pos_error, 'odom_rot': odom_rot_error})
                # print("position error", torch.max(torch.abs(position_error)))
                # print("rotation error", rotation_error)
                if self.training:
                    # print("training with gt pose")
                    pose = torch.cat([observations['odometry_emul']['agent_info']['xyz'][timestep],
                                      observations['odometry_emul']['agent_info']['rotation'][timestep].unsqueeze(-1)], dim=-1).unsqueeze(1)

            else:
                pose = torch.cat([observations['odometry_emul']['agent_info']['xyz'][timestep],
                                  observations['odometry_emul']['agent_info']['rotation'][timestep].unsqueeze(-1)], dim=-1).unsqueeze(1)

            map_update = self.project_depth_to_map(observations, pose.detach(), timestep, batch_size)
            all_maps[-1] += map_update
            all_maps[-1] = torch.clamp(all_maps[-1], min=0, max=1)
            prev_pose = pose

        
        # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()

        all_maps = all_maps[1:]
        all_maps = torch.stack(all_maps)

        
        # if all_maps.shape[0] > 1:
        #     # for i in range(all_maps.shape[0]):
        #     #     cv2.imwrite("/Users/karls/debug_images/train_all_map_{}_{}.png".format(self.step_count, i), all_maps[i, 0].detach().cpu().numpy())
        #     # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
        #     pass
        # else:
        #     for i in range(all_maps.shape[1]):
        #         cv2.imwrite("/Users/karls/debug_images/act_all_map_batch{}_{}.png".format(i, self.step_count), 
        #                     all_maps[-1, i, :, :, :3].detach().cpu().numpy()*256)
           
        
        
        # Transforms geocentric maps into egocentric maps
        ego_maps = self.transform_global_map_to_ego_map(all_maps, observations['odometry_emul']['agent_info'])

        self.step_count += 1
        
        pickup_bool = observations["pickedup_object"]
        after_pickup = pickup_bool == 1

        agent_mask = observations['object_mask_source'].clone()
        agent_mask[after_pickup] = observations['object_mask_destination'][after_pickup]

        arm_mask = observations['object_mask_kinect_source'].clone()
        arm_mask[after_pickup] = observations['object_mask_kinect_destination'][after_pickup]

        visual_observation = torch.cat([observations['depth_lowres'], observations['rgb_lowres'], agent_mask], dim=-1).float()
        visual_observation_encoding_body = compute_cnn_output(self.full_visual_encoder, visual_observation)

        visual_observation_arm = torch.cat([observations['depth_lowres_arm'], observations['rgb_lowres_arm'], arm_mask], dim=-1).float()
        visual_observation_encoding_arm = compute_cnn_output(self.full_visual_encoder_arm, visual_observation_arm)

        encoding_map = compute_cnn_output(self.full_visual_encoder_map, ego_maps)

        # From old model
        agent_distance_to_obj_source = observations['point_nav_emul_source'].clone()

        agent_distance_to_obj_destination = observations['point_nav_emul_destination'].clone()
        #TODO eventually change this and the following to only calculate embedding for the ones we want
        agent_distance_to_obj_embedding_source = self.body_pointnav_embedding(agent_distance_to_obj_source)
        agent_distance_to_obj_embedding_destination = self.body_pointnav_embedding(agent_distance_to_obj_destination)
        agent_distance_to_obj_embedding = agent_distance_to_obj_embedding_source
        agent_distance_to_obj_embedding[after_pickup] = agent_distance_to_obj_embedding_destination[after_pickup]


        arm_distance_to_obj_source = observations['arm_point_nav_emul_source'].clone()
        arm_distance_to_obj_destination = observations['arm_point_nav_emul_destination'].clone()
        #TODO eventually change this and the following to only calculate embedding for the ones we want
        arm_distance_to_obj_embedding_source = self.arm_pointnav_embedding(arm_distance_to_obj_source)
        arm_distance_to_obj_embedding_destination = self.arm_pointnav_embedding(arm_distance_to_obj_destination)
        arm_distance_to_obj_embedding = arm_distance_to_obj_embedding_source
        arm_distance_to_obj_embedding[after_pickup] = arm_distance_to_obj_embedding_destination[after_pickup]

        visual_observation_encoding = torch.cat([visual_observation_encoding_body, 
                                                 visual_observation_encoding_arm, 
                                                 agent_distance_to_obj_embedding, 
                                                 arm_distance_to_obj_embedding, 
                                                 encoding_map], dim=-1)

        # visual_observation_encoding = torch.cat([visual_observation_encoding_body, visual_observation_encoding_arm, ], dim=-1)

        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )



        actor_out_pickup = self.actor_pickup(x_out)
        critic_out_pickup = self.critic_pickup(x_out)

        actor_out_final = actor_out_pickup
        critic_out_final = critic_out_pickup

        actor_out = CategoricalDistr(logits=actor_out_final)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={'pose_errors': pose_errors}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)
        memory = memory.set_tensor("map", all_maps[-1].reshape(memory.tensor("map").shape))
        memory = memory.set_tensor("prev_pose", prev_pose.detach().reshape(memory.tensor("prev_pose").shape))

        return actor_critic_output, memory
