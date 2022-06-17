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
            visualize=False,
            ):
        super().__init__(action_space=action_space, observation_space=observation_space)


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
                                                [torch.sin(a),  torch.cos(a), pos[2]]] for a, pos in zip(ego_rotations, ego_xyz)], dtype=global_maps.dtype, device=global_maps.device)
        global_maps_reshaped = global_maps.reshape(-1, *global_maps.shape[2:]).permute(0, 3, 1, 2)

        affine_grid_world_to_ego = F.affine_grid(transform_world_to_ego, global_maps_reshaped.shape)
        ego_maps = F.grid_sample(global_maps_reshaped, affine_grid_world_to_ego)
        ego_maps = ego_maps.permute(0, 2, 3, 1).reshape(global_maps.shape)
        # print("egomaps", ego_maps.shape)
        # cv2.imwrite("/Users/karls/debug_images/act_all_map_{}_ego.png".format(self.step_count), ego_maps[-1, 0].detach().cpu().numpy()*256)
        return ego_maps

    def get_binned_map(
            self,
            frame: torch.FloatTensor,
            mask: torch.Tensor,
            camera: Dict, 
            timestep: int
        ) -> torch.FloatTensor:

        binned_updates = []
        for i in range(frame.shape[0]):
            if torch.sum(mask[i]) == 0:
                binned_updates.append(torch.zeros(*frame[i].shape, len(self.bins)+1, device=frame.device))
                continue

            camera_space_xyz = depth_frame_to_camera_space_xyz(frame[i], mask[i], fov=camera['fov'][timestep][i])
            world_points = camera_space_xyz_to_world_xyz(camera_space_xyzs=camera_space_xyz, 
                                                        camera_world_xyz=camera['xyz'][timestep][i], 
                                                        rotation=camera['rotation'][timestep][i].cpu(), 
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

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:

        batch_size = memory.tensor("map").shape[1]
        current_map = memory.tensor("map").reshape(batch_size, self.map_size, self.map_size, self.map_channels)
        # print("min max", torch.min(observations['depth_lowres']), torch.max(observations['depth_lowres']))

        # Builds geocentric maps
        all_maps = [current_map]
        for timestep in range(observations['depth_lowres'].shape[0]):
            all_maps.append(all_maps[-1].clone() * masks[timestep].reshape(batch_size, 1, 1, 1))
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
                
                binned_map_update = self.get_binned_map(reshaped_frame, valid_depths, camera, timestep)

                # Discards ceiling detections when updating the map
                all_maps[-1][:, :, :, :2] += binned_map_update[:, :, :, :2]

                # Adds the source object detections
                source_object_mask = torch.logical_and(valid_depths, source_mask[timestep].reshape(valid_depths.shape) > 0.0)
                if torch.sum(source_object_mask) > 0:
                    source_map_update = self.get_binned_map(reshaped_frame, source_object_mask, camera, timestep)
                    source_map_update = torch.sum(source_map_update, dim=-1)
                    all_maps[-1][:, :, :, 2] += source_map_update

                # Adds the target object detections
                destination_object_mask = torch.logical_and(valid_depths, destination_mask[timestep].reshape(valid_depths.shape) > 0.0)
                if torch.sum(destination_object_mask) > 0:
                    destination_map_update = self.get_binned_map(reshaped_frame, destination_object_mask, camera, timestep)
                    destination_map_update = torch.sum(destination_map_update, dim=-1)
                    all_maps[-1][:, :, :, 3] += destination_map_update

                # break
            
            # if True:
            #     # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
            #     robot_marker = torch.zeros_like(world_points)
            #     robot_marker[:, :, 1] = 20.0
            #     robot_marker[:, :, 0] = observations['odometry_emul']['agent_info']['xyz'][timestep, 0, 0]
            #     robot_marker[:, :, 2] = observations['odometry_emul']['agent_info']['xyz'][timestep, 0, 2]
            #     robot_marker += self.min_xyz
            #     all_maps[-1] += project_point_cloud_to_map(xyz_points = robot_marker,
            #                                                 bin_axis="y",
            #                                                 bins=self.bins,
            #                                                 map_size=self.map_size,
            #                                                 resolution_in_cm=self.map_resolution_cm,
            #                                                 flip_row_col=True)
                
            all_maps[-1] = torch.clamp(all_maps[-1], min=0, max=1)

        

        
        # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
        # plt.figure()
        # agent_poses_numpy = torch.stack(self.agent_poses)
        # camera_poses_numpy = torch.stack(self.camera_poses)
        # plt.scatter(agent_poses_numpy[:, :, :, 0], agent_poses_numpy[:, :, :, 2], label="agent")
        # plt.scatter(camera_poses_numpy[:, :, :, 0], camera_poses_numpy[:, :, :, 2], label="camera")
        # plt.legend()
        # plt.savefig("/Users/karls/agent_poses_{}.png".format(self.step_count))
        all_maps = all_maps[1:]
        all_maps = torch.stack(all_maps)

        
        # if all_maps.shape[0] > 1:
        #     # for i in range(all_maps.shape[0]):
        #     #     cv2.imwrite("/Users/karls/debug_images/train_all_map_{}_{}.png".format(self.step_count, i), all_maps[i, 0].detach().cpu().numpy())
        #     # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
        #     pass
        # else:
        #     for i in range(all_maps.shape[1]):
        #         cv2.imwrite("/Users/karls/debug_images/act_all_map_batch{}_{}.png".format(i, self.step_count), all_maps[-1, i, :, :, :3].detach().cpu().numpy()*256)
        #     # cv2.imwrite("/Users/karls/debug_images/act_all_map_{}_rgb_front.png".format(self.step_count), observations['rgb_lowres'][-1, 0].cpu().numpy() * 50+ 100 )
        #     # cv2.imwrite("/Users/karls/debug_images/act_all_map_{}_rgb_arm.png".format(self.step_count), observations['rgb_lowres_arm'][-1, 0].cpu().numpy() * 50+ 100 )
        #     # cv2.imwrite("/Users/karls/debug_images/act_all_map_{}_depth_front.png".format(self.step_count), observations['depth_lowres'][-1, 0].cpu().numpy() * 50+ 100 )
        #     # cv2.imwrite("/Users/karls/debug_images/act_all_map_{}_depth_arm.png".format(self.step_count), observations['depth_lowres_arm'][-1, 0].cpu().numpy() * 50+ 100 )

        
        
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

        #visual_observation_encoding = torch.cat([visual_observation_encoding_body, visual_observation_encoding_arm, encoding_map], dim=-1)


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

        visual_observation_encoding = torch.cat([visual_observation_encoding_body, visual_observation_encoding_arm, agent_distance_to_obj_embedding, arm_distance_to_obj_embedding, encoding_map], dim=-1)

        # visual_observation_encoding = torch.cat([visual_observation_encoding_body, visual_observation_encoding_arm, ], dim=-1)

        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )


        # I think we need two model one for pick up and one for drop off

        actor_out_pickup = self.actor_pickup(x_out)
        critic_out_pickup = self.critic_pickup(x_out)

        actor_out_final = actor_out_pickup
        critic_out_final = critic_out_pickup

        actor_out = CategoricalDistr(logits=actor_out_final)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={}
        )

        memory = memory.set_tensor("rnn", rnn_hidden_states)
        memory = memory.set_tensor("map", all_maps[-1].reshape(memory.tensor("map").shape))

        return actor_critic_output, memory
