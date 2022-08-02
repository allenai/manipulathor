import torch
import torch.nn as nn

import isdf
from isdf.modules import trainer
from isdf.datasets.data_util import FrameData

class SdfTrainerMultiprocessingWrapper():
    def __init__(self,
                 index,
                 device):
        self.device = device
        self.batch = index

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



        self.map = self.init_new_map()
    
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
        self.step_count = 0
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
        self.step_count += 1
        return all_data
    
    def update_map(self, observations, masks):

        num_timesteps = observations['depth_lowres'].shape[0]

        step_scale = 10

        outputs = []
        for timestep in range(num_timesteps):
            if masks[timestep][self.batch] == 0:
                self.trainer = self.init_new_map()
            
            frame_data = self.format_frame_data(observations, timestep, self.batch)
            self.trainer.add_frame(frame_data)

            if masks[timestep][self.batch] == 0:
                self.trainer.last_is_keyframe = True
                self.trainer.optim_frames = 200
            else:
                T_WC = frame_data.T_WC_batch#[-1].unsqueeze(0)
                depth_gt = frame_data.depth_batch#[-1].unsqueeze(0)
                dirs_c = frame_data.dirs_c_batch#[-1].unsqueeze(0)
                self.trainer.last_is_keyframe = self.trainer.is_keyframe(T_WC, depth_gt, dirs_c)
                if self.trainer.last_is_keyframe:
                    self.trainer.optim_frames = self.trainer.iters_per_kf

            with torch.enable_grad():
                losses, _ = self.trainer.step()

                for i in range(self.trainer.optim_frames // step_scale):
                    losses, _ = self.trainer.step()

            output = self.trainer.compute_slices_rotated(
                    self.sampling_pc,
                    observations['odometry_emul']['agent_info']['xyz'][timestep, self.batch],
                    observations['odometry_emul']['agent_info']['rotation'][timestep, self.batch]
            )

            outputs.append(output.permute(1, 2, 0))
        outputs = torch.stack(outputs).detach()
        return outputs


def start_process(index, device, obs_queue, output_queue):
    print("starting process")
    wrapper = SdfTrainerMultiprocessingWrapper(index, device)
    print("inited wrapper")
    while True:
        print("waiting for obs")
        obs_and_masks = obs_queue.get()
        print("got obs", index)
        observations = obs_and_masks[0]
        masks = obs_and_masks[1]

        output = wrapper.update_map(observations, masks)
        output_queue.put((index, output))
        print("sent output", index)