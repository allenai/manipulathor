"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
import platform
from datetime import datetime
from typing import Tuple, Optional, Dict
import time

import cv2
import gym
import torch
import torch.nn.functional as F
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
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

import isdf
from isdf.modules import trainer
from isdf.datasets.data_util import FrameData
from isdf.datasets.sdf_util import get_colormap

from manipulathor_utils.debugger_util import ForkedPdb
from utils.model_utils import LinearActorHeadNoCategory
from utils.hacky_viz_utils import hacky_visualization
from utils.stretch_utils.stretch_thor_sensors import check_for_nan_visual_observations

from manipulathor_baselines.stretch_bring_object_baselines.models.network_encoder_model import NetworkEncoderModel

class StretchObjectDisplacementISDFModel(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model for preddistancenav task.

    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        obj_state_embedding_size=512,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type="GRU",
        teacher_forcing=1,
        use_odom_pose=False,
        visualize=False,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.visualize = visualize

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size
        self.use_odom_pose = use_odom_pose

        # sensor_names = self.observation_space.spaces.keys()
        network_args = {'input_channels': 5,
                        'layer_channels': [32, 64, 32],
                        'kernel_sizes': [(8, 8), (4, 4), (3, 3)],
                        'strides': [(4, 4), (2, 2), (1, 1)],
                        'paddings': [(0, 0), (0, 0), (0, 0)],
                        'dilations': [(1, 1), (1, 1), (1, 1)],
                        'output_height': 24,
                        'output_width': 24,
                        'output_channels': 512,
                        'flatten': True,
                        'output_relu': True}
        self.full_visual_encoder = make_cnn(**network_args)
        network_args = {'input_channels': 5,
                        'layer_channels': [32, 64, 32],
                        'kernel_sizes': [(8, 8), (4, 4), (3, 3)],
                        'strides': [(4, 4), (2, 2), (1, 1)],
                        'paddings': [(0, 0), (0, 0), (0, 0)],
                        'dilations': [(1, 1), (1, 1), (1, 1)],
                        'output_height': 24,
                        'output_width': 24,
                        'output_channels': 512,
                        'flatten': True,
                        'output_relu': True}
        self.full_visual_encoder_arm = make_cnn(**network_args)

        

        # self.detection_model = ConditionalDetectionModel()
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
        num_rnn_inputs = 5
        if use_odom_pose:
            self.odom_pose_embedding = nn.Sequential(
                nn.Linear(3, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 512),
            )
            num_rnn_inputs += 1

        self.state_encoder = RNNStateEncoder(
            512 * num_rnn_inputs, #TODO this might be too big, maybe combine visual encodings and pointnav encodings first
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pickup = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_pickup = LinearCriticHead(self._hidden_size)

        self.device = None
        self.sdf_trainers = []
        self.no_norm = False
        self.config_file = "../iSDF/isdf/train/configs/thor_live.json"
        if self.no_norm:
            self.config_file = "../iSDF/isdf/train/configs/thor_live_no_norm.json"

        self.max_depth = 5.0
        # Only apply min depth to the arm camera to filter out pictures of the arm
        self.min_depth_arm = 1.0

        # Need to divide by range_dist as it will scale the grid which
        # is created in range = [-1, 1]
        # Also divide by 0.9 so extents are a bit larger than gt mesh
        #self.bounds_transform = torch.eye(4)
        self.bounds_transform = torch.tensor([[-2.06479118e-23,  1.00000000e+00,  6.12323400e-17,  9.52677835e-01],
             [ 1.00000000e+00, -4.23140135e-39,  3.37205989e-07,  1.48196393e+00],                                          
              [ 3.37205989e-07,  6.12323400e-17, -1.00000000e+00,  1.66513429e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        self.bounds_transform = torch.tensor([[0, 1, 0, 0],#0.95],
                                              [1, 0, 0, 1.48],
                                              [0, 0, -1, 0],#1.67],
                                              [0, 0, 0, 1.0]])

        self.grid_dim = 112#224
        grid_range = [-1.0, 1.0]
        range_dist = grid_range[1] - grid_range[0]
        # Gives the same size of map as the grid based approach
        #bounds_extents = torch.tensor([[1.7362, 4.0325, 7.2216]])#torch.tensor([5.6, 1.5, 5.6])
        self.scene_scale = torch.tensor([1.7362, 4.0325, 7.2216])
        self.scene_scale = torch.tensor([1.7362, 5.6, 5.6])
        self.scene_scale = torch.tensor([1.7362, 4.0, 4.0])
        #self.scene_scale = bounds_extents / (range_dist * 0.9)
        self.inv_scene_scale = 1. / self.scene_scale

        grid_pc = isdf.geometry.transform.make_3D_grid(
            grid_range,
            self.grid_dim,
            self.device if self.device is not None else torch.device("cpu"),
            transform=self.bounds_transform,
            scale=self.scene_scale,
        )
        grid_pc = grid_pc.view(-1, 3)
        self.up_ix = 0

        n_slices = 6
        z_ixs = torch.linspace(30, self.grid_dim - 30, n_slices)
        z_ixs = torch.round(z_ixs).long()
        z_ixs = z_ixs.to(grid_pc.device)

        self.sampling_pc = grid_pc.reshape(self.grid_dim, self.grid_dim, self.grid_dim, 3)
        self.sampling_pc = torch.index_select(self.sampling_pc, self.up_ix, z_ixs)

        self.camera_fx_fy = 162.96101
        self.camera_arm_fx_fy = 112
        self.cx_cy = 112
        self.height_width = 224

        self.dirs_c_batch_camera = isdf.geometry.transform.ray_dirs_C(
                1,
                self.height_width, self.height_width,
                self.camera_fx_fy, self.camera_fx_fy,
                self.cx_cy, self.cx_cy,
                self.device,
                depth_type="z")

        self.dirs_c_batch_camera_arm = isdf.geometry.transform.ray_dirs_C(
                1,
                self.height_width, self.height_width,
                self.camera_arm_fx_fy, self.camera_arm_fx_fy,
                self.cx_cy, self.cx_cy,
                self.device,
                depth_type="z")

        self.encode_sdf_net_params = False
        if not self.encode_sdf_net_params:
            network_args = {'input_channels': 6, #18,
                            'layer_channels': [32, 64, 32],
                            'kernel_sizes': [(8, 8), (4, 4), (3, 3)],
                            'strides': [(4, 4), (2, 2), (1, 1)],
                            'paddings': [(0, 0), (0, 0), (0, 0)],
                            'dilations': [(1, 1), (1, 1), (1, 1)],
                            'output_height': 10,
                            'output_width': 10,
                            #'output_height': 24,
                            #'output_width': 24,
                            'output_channels': 512,
                            'flatten': True,
                            'output_relu': True}
            self.map_encoder = make_cnn(**network_args)
        else:
            self.sdf_encoder = NetworkEncoderModel(self.init_new_map().sdf_map)


        self.multiprocessing = False
        if self.multiprocessing:
            self.obs_queue = torch.multiprocessing.Queue()
            self.output_queue = torch.multiprocessing.Queue()

        self.cmap = get_colormap()

        self.step_count = 0

        self.train()



        self.starting_time = datetime.now().strftime("{}_%m_%d_%Y_%H_%M_%S_%f".format(self.__class__.__name__))



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
            )
        )

    def init_new_map(self):
        sdf_map = trainer.Trainer(
            self.device,
            self.config_file,
            chkpt_load_file=None,
            incremental=True,
        )
        #sdf_map.grid_pc = self.grid_pc.clone().to(self.device)
        sdf_map.grid_dim = self.grid_dim
        sdf_map.up_ix = self.up_ix
        sdf_map.scene_scale = self.scene_scale
        sdf_map.grid_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        sdf_map.up_aligned = True
        sdf_map.bounds_transform_np = self.bounds_transform.cpu().numpy()
        #from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
        #print("created new map")
        return sdf_map

    def format_frame_data(self, observations, timestep, batch):
        all_data = FrameData()

        for camera_name in ['', '_arm']:
            depth = observations['depth_lowres' + camera_name][timestep, batch].reshape(1, 224, 224)
            depth[depth > self.max_depth] = 0
            if camera_name == '_arm':
                mask = observations['agent_mask_arm'][timestep, batch].unsqueeze(0)
                dilation_size = 1
                dilation_mask = torch.ones((1, 1, dilation_size*2+1, dilation_size*2+1), device=mask.device)
                dilated_mask = nn.functional.conv2d(mask.unsqueeze(0).to(torch.float32), dilation_mask, padding=dilation_size)[0].to(torch.bool)
                depth[dilated_mask] = 0
            transform = observations['odometry_emul']['camera_info']['camera'+camera_name]['gt_transform'][timestep, batch]
            # First element of scene_scale is the virtical scale
            #transform[0, -1] /= self.scene_scale[1]
            #transform[2, -1] /= self.scene_scale[2]
            transform = torch.linalg.inv(transform).unsqueeze(0)

            if camera_name == '':
                fx = self.camera_fx_fy
                fy = self.camera_fx_fy
                dirs_c_batch = self.dirs_c_batch_camera.clone()
            else:
                fx = self.camera_arm_fx_fy
                fy = self.camera_arm_fx_fy
                dirs_c_batch = self.dirs_c_batch_camera_arm.clone()
            #cx = 112
            #cy = 112
            #height = 224
            #width = 224
            cx = self.cx_cy
            cy = self.cx_cy
            height = self.height_width
            width = self.height_width

            frame_id = self.step_count + timestep
            if camera_name == '_arm':
                frame_id *= -1
            camera = FrameData(
                frame_id=torch.tensor([frame_id]),
                im_batch=observations['rgb_lowres' + camera_name][timestep, batch].unsqueeze(0),
                depth_batch=depth,
                T_WC_batch=transform,
                T_WC_batch_np=transform.cpu().numpy(),
                dirs_c_batch=dirs_c_batch,
            )

            if not self.no_norm:
                pc = isdf.geometry.transform.pointcloud_from_depth_torch(
                    depth[0], fx, fy, cx, cy)
                normals = isdf.geometry.transform.estimate_pointcloud_normals(pc)
                camera.normal_batch = normals[None, :]

            all_data.add_frame_data(camera, replace=False)
        return all_data

    def update_maps(self, observations, masks):
        num_timesteps = observations['depth_lowres'].shape[0]
        num_batches = observations['depth_lowres'].shape[1]
        all_outputs = []

        step_scale = 10

        init_time = 0.0
        data_time = 0.0
        data_1 = 0.0
        is_keyframe_time = 0.0
        optim_time = 0.0
        render_time = 0.0

        # Mapping must always use gradients to update the sdf
        # even if the rest of the model is evaluating
        with torch.enable_grad():
            for timestep in range(num_timesteps):
                batch_outputs = []
                for batch in range(num_batches):
                    if batch > len(self.sdf_trainers) - 1:
                        st = time.perf_counter()
                        self.sdf_trainers.append(self.init_new_map())
                        et = time.perf_counter()
                        init_time += (et - st)
                    elif  masks[timestep][batch] == 0:
                        st = time.perf_counter()
                        self.sdf_trainers[batch] = self.init_new_map()
                        et = time.perf_counter()
                        init_time += (et - st)
                    
                    st = time.perf_counter()
                    frame_data = self.format_frame_data(observations, timestep, batch)
                    et = time.perf_counter()
                    data_1 += (et - st)
                    st = time.perf_counter()
                    self.sdf_trainers[batch].add_frame(frame_data)
                    et = time.perf_counter()
                    data_time += (et - st)
                    if masks[timestep][batch] == 0:
                        self.sdf_trainers[batch].last_is_keyframe = True
                        self.sdf_trainers[batch].optim_frames = 200
                    else:
                        st = time.perf_counter()
                        # Only checks on one camera
                        T_WC = frame_data.T_WC_batch#[-1].unsqueeze(0)
                        depth_gt = frame_data.depth_batch#[-1].unsqueeze(0)
                        dirs_c = frame_data.dirs_c_batch#[-1].unsqueeze(0)
                        self.sdf_trainers[batch].last_is_keyframe = self.sdf_trainers[batch].is_keyframe(T_WC, depth_gt, dirs_c)
                        if self.sdf_trainers[batch].last_is_keyframe:
                            self.sdf_trainers[batch].optim_frames = self.sdf_trainers[batch].iters_per_kf
                        et = time.perf_counter()
                        is_keyframe_time += (et - st)

                    st = time.perf_counter()
                    losses, _ = self.sdf_trainers[batch].step()

                    for i in range(self.sdf_trainers[batch].optim_frames // step_scale):
                        losses, _ = self.sdf_trainers[batch].step()
                    et = time.perf_counter()
                    optim_time += (et - st)

                    if not self.encode_sdf_net_params:
                        st = time.perf_counter()
                        output = self.sdf_trainers[batch].compute_slices_rotated(
                                self.sampling_pc,
                                observations['odometry_emul']['agent_info']['xyz'][timestep, batch],
                                observations['odometry_emul']['agent_info']['rotation'][timestep, batch]
                        )

                        #output = self.sdf_trainers[batch].compute_slices(draw_cams=True)
                        slices = output.permute(1, 2, 0)
                        et = time.perf_counter()
                        render_time += (et - st)
                        #from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
                        #import cv2
                        #for s in range(len(output['pred_sdf'])):
                        #    cv2.imwrite("../debug_images/pred_{}_ts{}_b{}_s{}.png".format(self.step_count, timestep, batch, s),
                        #            output["pred_sdf"][s][..., ::-1])
                        #import numpy as np
                        #slices = torch.from_numpy(np.concatenate(output['pred_sdf'], axis=-1)).to(self.device).to(torch.float32)
                        #slices = slices[:224, :224]
                        #slices = torch.zeros(224, 224, 18).to(self.device)

                        batch_outputs.append(slices)
                if self.encode_sdf_net_params:
                    st = time.perf_counter()
                    batch_outputs = self.sdf_encoder([trainer.sdf_map for trainer in self.sdf_trainers])
                    et = time.perf_counter()
                    render_time += (et - st)
                else:
                    batch_outputs = torch.stack(batch_outputs)
                all_outputs.append(batch_outputs)
        all_outputs = torch.stack(all_outputs)
        all_outputs = all_outputs.detach()
        # print("init", init_time)
        # print("data1", data_1)
        # print("data", data_time)
        # print("is keyframe", is_keyframe_time)
        # print("optim", optim_time)
        # print("render", render_time)
        return all_outputs

    def transform_global_map_to_ego_map(
        self,
        global_maps: torch.FloatTensor,
        agent_info: Dict,
    ) -> torch.FloatTensor:

        ego_rotations = -torch.deg2rad(agent_info['rotation'].reshape(-1))
        map_resolution_cm = 5.0
        map_size = 224
        ego_xyz = agent_info['xyz'].reshape(-1, 3) / (map_resolution_cm / 100.) / map_size * 2.0
        transform_world_to_ego = torch.tensor([[[torch.cos(a), -torch.sin(a), pos[0]],
                                                [torch.sin(a),  torch.cos(a), pos[2]]] for a, pos in zip(ego_rotations, ego_xyz)], 
                                                dtype=global_maps.dtype, device=global_maps.device)
        global_maps_reshaped = global_maps.reshape(-1, *global_maps.shape[2:]).permute(0, 3, 1, 2)
        affine_grid_world_to_ego = F.affine_grid(transform_world_to_ego, global_maps_reshaped.shape)
        ego_maps = F.grid_sample(global_maps_reshaped, affine_grid_world_to_ego, align_corners=False)
        # from manipulathor_utils.debugger_util import ForkedPdb; ForkedPdb().set_trace()
        ego_maps = ego_maps.permute(0, 2, 3, 1).reshape(global_maps.shape)
        return ego_maps

    def update_maps_multiprocessing(self, observations, masks):
        from manipulathor_baselines.stretch_bring_object_baselines.models.sdf_trainer_multiprocessing_wrapper import start_process
        num_batches = observations['depth_lowres'].shape[1]
        for i in range(num_batches):
            if i > len(self.sdf_trainers) - 1:
                p = torch.multiprocessing.Process(target=start_process,
                                                  args=(i, self.device, self.obs_queue, self.output_queue))
                p.start()
                self.sdf_trainers.append(p)
            
        for i in range(num_batches):
            self.obs_queue.put((observations, masks))
            print("sent obs", i)
    
        output_unsorted = []
        for i in range(num_batches):
            output_unsorted.append(self.output_queue.get())
            print("got output", output_unsorted[-1][0])
        output = [o[1] for o in sorted(output_unsorted)]
        
        output = torch.stack(output)
        output = output.permute(1, 0, 2, 3, 4)
        return output



    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """
        if self.step_count == 0:
            self.start_time = time.perf_counter()
        else:
            end_time = time.perf_counter()
            fps = self.step_count / (end_time - self.start_time)
            #print("fps", fps, fps*8, "frame", self.step_count)

        if self.device is None:
            self.device = self.full_visual_encoder_arm[0].bias.device
            self.sampling_pc = self.sampling_pc.to(self.device)
            self.dirs_c_batch_camera = self.dirs_c_batch_camera.to(self.device)
            self.dirs_c_batch_camera_arm = self.dirs_c_batch_camera_arm.to(self.device)

        mapping_start_time = time.perf_counter()
        if self.multiprocessing:
            all_maps = self.update_maps_multiprocessing(observations, masks)
        else:
            all_maps = self.update_maps(observations, masks)
        print("all_maps", all_maps.shape)
        #all_maps = self.transform_global_map_to_ego_map(all_maps, observations['odometry_emul']['agent_info'])
        if all_maps.shape[0] > 1:
            end_time = time.perf_counter()
            print("mapping_time", end_time - mapping_start_time)
        if self.step_count % 100 == 0 and self.device.index == 0:
            if all_maps.shape[0] == 1:
                for batch in range(all_maps.shape[1]):
                    for s in range(all_maps.shape[-1]):
                        if s != 3:
                            continue
                        im = all_maps[0, batch, :, :, s].cpu().numpy()
                        im = self.cmap.to_rgba(im)[:, :, :3] * 255
                        px = im.shape[0] //2
                        py = im.shape[1] //2
                        im[px-1:px+1, py-1:py+1, :] = (0, 255, 0)
                        im = im[..., ::-1]
                        cv2.imwrite("../debug_images/pred_{}_b{}_s{}.png".format(self.step_count, batch, s), im)
                    print("saved images", self.step_count)
                    break
        if not self.encode_sdf_net_params:
            map_embedding = compute_cnn_output(self.map_encoder, all_maps)
        else:
            map_embedding = all_maps
#        map_embedding = torch.zeros((observations['rgb_lowres'].shape[0], 
#                                     observations['rgb_lowres'].shape[1],
#                                     512), device=observations['rgb_lowres'].device)

        #we really need to switch to resnet now that visual features are actually important
        pickup_bool = observations["pickedup_object"]
        after_pickup = pickup_bool == 1

        agent_mask = observations['object_mask_source'].clone()
        agent_mask[after_pickup] = observations['object_mask_destination'][after_pickup]

        arm_mask = observations['object_mask_kinect_source'].clone()
        arm_mask[after_pickup] = observations['object_mask_kinect_destination'][after_pickup]

        #TODO remove these after issue is resolved
        observations['depth_lowres'] = check_for_nan_visual_observations(observations['depth_lowres'], where_it_occured='depth_lowres')
        observations['rgb_lowres'] = check_for_nan_visual_observations(observations['rgb_lowres'], where_it_occured='rgb_lowres')
        observations['depth_lowres_arm'] = check_for_nan_visual_observations(observations['depth_lowres_arm'], where_it_occured='depth_lowres_arm')
        observations['rgb_lowres_arm'] = check_for_nan_visual_observations(observations['rgb_lowres_arm'], where_it_occured='rgb_lowres_arm')
        arm_mask = check_for_nan_visual_observations(arm_mask, where_it_occured='arm_mask')
        agent_mask = check_for_nan_visual_observations(agent_mask, where_it_occured='agent_mask')

        visual_observation = torch.cat([observations['depth_lowres'], observations['rgb_lowres'], agent_mask], dim=-1).float()

        visual_observation_encoding_body = compute_cnn_output(self.full_visual_encoder, visual_observation)

        visual_observation_arm = torch.cat([observations['depth_lowres_arm'], observations['rgb_lowres_arm'], arm_mask], dim=-1).float()
        visual_observation_arm = check_for_nan_visual_observations(visual_observation_arm)
        visual_observation_encoding_arm = compute_cnn_output(self.full_visual_encoder_arm, visual_observation_arm)
        # ForkedPdb().set_trace()



        # arm_distance_to_obj_source = observations['point_nav_real_source'].copy()
        # arm_distance_to_obj_destination = observations['point_nav_real_destination'].copy()
        # arm_distance_to_obj_source_embedding = self.pointnav_embedding(arm_distance_to_obj_source)
        # arm_distance_to_obj_destination_embedding = self.pointnav_embedding(arm_distance_to_obj_destination)
        # pointnav_embedding = arm_distance_to_obj_source_embedding
        # pointnav_embedding[after_pickup] = arm_distance_to_obj_destination_embedding[after_pickup]

        #TODO remove
        observations['point_nav_emul_source'] = check_for_nan_visual_observations(observations['point_nav_emul_source'], where_it_occured='point_nav_emul_source')
        observations['point_nav_emul_destination'] = check_for_nan_visual_observations(observations['point_nav_emul_destination'], where_it_occured='point_nav_emul_destination')
        observations['arm_point_nav_emul_source'] = check_for_nan_visual_observations(observations['arm_point_nav_emul_source'], where_it_occured='arm_point_nav_emul_source')
        observations['arm_point_nav_emul_destination'] = check_for_nan_visual_observations(observations['arm_point_nav_emul_destination'], where_it_occured='arm_point_nav_emul_destination')

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


        if self.use_odom_pose:
            odom_observation = torch.stack([observations['odometry_emul']['agent_info']['xyz'][:, :, 0],
                                            observations['odometry_emul']['agent_info']['xyz'][:, :, 2],
                                            observations['odometry_emul']['agent_info']['rotation']], dim=-1).float()
            odom_embedding = self.odom_pose_embedding(odom_observation)

            visual_observation_encoding = torch.cat([visual_observation_encoding_body,
                                                     visual_observation_encoding_arm,
                                                     agent_distance_to_obj_embedding,
                                                     arm_distance_to_obj_embedding,
                                                     map_embedding,
                                                     odom_embedding], dim=-1)
        else:
            visual_observation_encoding = torch.cat([visual_observation_encoding_body,
                                                     visual_observation_encoding_arm,
                                                     agent_distance_to_obj_embedding,
                                                     arm_distance_to_obj_embedding,
                                                     map_embedding], dim=-1)

        x_out, rnn_hidden_states = self.state_encoder(
            visual_observation_encoding, memory.tensor("rnn"), masks
        )


        # I think we need two model one for pick up and one for drop off

        actor_out_final = self.actor_pickup(x_out)
        critic_out_final = self.critic_pickup(x_out)

        actor_out_final = check_for_nan_visual_observations(actor_out_final, where_it_occured='actor_out_final')
        #actor_out_final[:, :, :] = -100000
        #actor_out_final[:, :, 7] = 100000
        actor_out = CategoricalDistr(logits=actor_out_final)

        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out_final, extras={}
        )

        self.step_count += 1

        memory = memory.set_tensor("rnn", rnn_hidden_states)


        if self.visualize:
            # if arm_distance_to_obj_source.shape[0] == 1 and arm_distance_to_obj_source.shape[1] == 1:
            #     arm_distances = arm_distance_to_obj_source
            #     arm_distances[after_pickup] = arm_distance_to_obj_destination
            #     agent_distances = agent_distance_to_obj_source
            #     agent_distances[after_pickup] = agent_distance_to_obj_destination
            #     print('arm_distances', arm_distances, 'agent_distances', agent_distances)

            source_object_mask = observations['object_mask_source']
            destination_object_mask = observations['object_mask_destination']
            intel_mask = source_object_mask
            intel_mask[after_pickup] = destination_object_mask[after_pickup]


            source_object_mask_kinect = observations['object_mask_kinect_source']
            destination_object_mask_kinect = observations['object_mask_kinect_destination']
            kinect_mask = source_object_mask_kinect
            kinect_mask[after_pickup] = destination_object_mask_kinect[after_pickup]

            distances = torch.cat([observations['point_nav_emul_source'],observations['arm_point_nav_emul_source']], dim=-1)
            # distances = observations['point_nav_emul_source']

            agent_distance_vector_to_viz =  observations['point_nav_emul_source'].clone()
            agent_distance_vector_to_viz[after_pickup] = observations['point_nav_emul_destination'][after_pickup].clone()

            arm_distance_vector_to_viz =  observations['arm_point_nav_emul_source'].clone()
            arm_distance_vector_to_viz[after_pickup] = observations['arm_point_nav_emul_destination'][after_pickup].clone()

            distance_vector_to_viz = dict(arm_dist=arm_distance_vector_to_viz, agent_dist=agent_distance_vector_to_viz)

            hacky_visualization(observations, 
                                object_mask=intel_mask, 
                                gt_mask=kinect_mask,
                                base_directory_to_right_images=self.starting_time,
                                text_to_write=distances,
                                distance_vector_to_viz=distance_vector_to_viz)




        return (
            actor_critic_output,
            memory,
        )

