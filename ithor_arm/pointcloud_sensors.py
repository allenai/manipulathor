import copy
import datetime
import os
from typing import Any

from typing import Optional, Sequence, Dict

from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import (
    depth_frame_to_world_space_xyz,
    project_point_cloud_to_map,
)

import ai2thor.controller
import gym
import gym.spaces
import numpy as np
import torch

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment


def show_3d(things_to_show, map_size = None, additional_tag=''):
    # import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if len(things_to_show.shape) == 3:
        x, y, z = things_to_show[:,:,0], things_to_show[:,:,1], things_to_show[:,:,2]
    elif len(things_to_show.shape) == 2:
        x, y, z = things_to_show[:,0], things_to_show[:,1], things_to_show[:,2]
    if map_size is not None:
        ax.axes.set_xlim3d(left=0, right=map_size[0])
        ax.axes.set_ylim3d(bottom=0, top=map_size[1])
        ax.axes.set_zlim3d(bottom=0, top=map_size[2])
    ax.scatter3D(x, y, z, cmap='Greens')
    imagename = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S.%f") + additional_tag +'.png'
    base_dir = '/Users/kianae/Desktop/pointcloud_viz'
    os.makedirs(base_dir, exist_ok=True)
    plt.savefig(os.path.join(base_dir, imagename))

class KianaBinnedPointCloudMapTHORSensor(
    Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]
):
    def __init__(
            self,
            fov: float,
            # vision_range_in_cm: int,
            map_size_in_cm: int,
            resolution_in_cm: int,
            map_range_sensor: Sensor,
            mask_sensor: Sensor,
            type: str,
            # height_bins: Sequence[float] = (0.02, 2),
            height_bins: Sequence[float] = tuple([i * 0.2 for i in range(0, 10)]),
            ego_only: bool = True,
            uuid: str = "binned_pc_map",
            **kwargs: Any,
    ):
        self.fov = fov
        # self.vision_range_in_cm = vision_range_in_cm
        self.map_size_in_cm = map_size_in_cm
        self.resolution_in_cm = resolution_in_cm
        self.height_bins = height_bins
        self.ego_only = ego_only
        self.mask_sensor = mask_sensor

        uuid = uuid + '_' + type

        self.binned_pc_map_builder = KianaBinnedPointCloudMapBuilder(
            fov=fov,
            # vision_range_in_cm=vision_range_in_cm,
            map_size_in_cm=map_size_in_cm,
            resolution_in_cm=resolution_in_cm,
            height_bins=height_bins,
        )

        map_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=self.binned_pc_map_builder.binned_point_cloud_map.shape,
            dtype=np.float32,
        )

        space_dict = {
            "egocentric_update": map_space,
        }
        if not ego_only:
            space_dict["allocentric_update"] = copy.deepcopy(map_space)
            space_dict["map"] = copy.deepcopy(map_space)
            space_dict["map_with_agent"] = copy.deepcopy(map_space)

        observation_space = gym.spaces.Dict(space_dict)
        super().__init__(**prepare_locals_for_super(locals()))

        self.map_range_sensor = map_range_sensor

    @property
    def device(self):
        return self.binned_pc_map_builder.device

    @device.setter
    def device(self, val: torch.device):
        self.binned_pc_map_builder.device = torch.device(val)

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[Task[RoboThorEnvironment]],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        e = env.controller.last_event
        metadata = e.metadata

        mask = self.mask_sensor.get_observation(env, task).astype(bool).squeeze(-1)
        depth_frame_for_target_obj = e.depth_frame.copy()
        depth_frame_for_target_obj[~mask] = -1

        if task.num_steps_taken() == 0:
            xyz_ranges_dict = self.map_range_sensor.get_observation(env=env, task=task)
            self.binned_pc_map_builder.reset(
                min_xyz=np.array(
                    [
                        xyz_ranges_dict["x_range"][0],
                        0, # TODO xyz_ranges_dict["y_range"][0],
                        xyz_ranges_dict["z_range"][0],
                    ]
                )
            )


        map_dict = self.binned_pc_map_builder.update(depth_frame=depth_frame_for_target_obj,camera_xyz=np.array([metadata["cameraPosition"][k] for k in ["x", "y", "z"]]),camera_rotation=metadata["agent"]["rotation"]["y"],camera_horizon=metadata["agent"]["cameraHorizon"],)

        return {k: map_dict[k] for k in self.observation_space.spaces.keys()}

def rotate_points_to_agent(world_space_point_cloud, device, camera_xyz, camera_rotation, map_size_in_cm):
    recentered_point_cloud = world_space_point_cloud - (
            torch.FloatTensor([1.0, 0.0, 1.0]).to(device) * camera_xyz
    ).reshape((1, 1, 3))
    # Rotate the cloud so that positive-z is the direction the agent is looking
    theta = (
            np.pi * camera_rotation / 180
    )  # No negative since THOR rotations are already backwards
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_transform = torch.FloatTensor(
        [
            [cos_theta, 0, -sin_theta],
            [0, 1, 0],  # unchanged
            [sin_theta, 0, cos_theta],
        ]
    ).to(device)
    rotated_point_cloud = recentered_point_cloud @ rotation_transform.T
    xoffset = (map_size_in_cm / 100) / 2
    agent_centric_point_cloud = rotated_point_cloud + torch.FloatTensor(
        [xoffset, 0, 0]
    ).to(device)
    return agent_centric_point_cloud

class KianaBinnedPointCloudMapBuilder(object):
    """Class used to iteratively construct a map of "free space" based on input
    depth maps (i.e. pointclouds).

    Adapted from https://github.com/devendrachaplot/Neural-SLAM

    This class can be used to (iteratively) construct a metric map of free space in an environment as
    an agent moves around. After every step the agent takes, you should call the `update` function and
    pass the agent's egocentric depth image along with the agent's new position. This depth map will
    be converted into a pointcloud, binned along the up/down axis, and then projected
    onto a 3-dimensional tensor of shape (HxWxC) whose where HxW represent the ground plane
    and where C equals the number of bins the up-down coordinate was binned into. This 3d map counts the
    number of points in each bin. Thus a lack of points within a region can be used to infer that
    that region is free space.

    # Attributes

    fov : FOV of the camera used to produce the depth images given when calling `update`.
    vision_range_in_map_units : The maximum distance (in number of rows/columns) that will
        be updated when calling `update`, points outside of this map vision range are ignored.
    map_size_in_cm : Total map size in cm.
    resolution_in_cm : Number of cm per row/column in the map.
    height_bins : The bins used to bin the up-down coordinate (for us the y-coordinate). For example,
        if `height_bins = [0.1, 1]` then
        all y-values < 0.1 will be mapped to 0, all y values in [0.1, 1) will be mapped to 1, and
        all y-values >= 1 will be mapped to 2.
        **Importantly:** these y-values will first be recentered by the `min_xyz` value passed when
        calling `reset(...)`.
    device : A `torch.device` on which to run computations. If this device is a GPU you can potentially
        obtain significant speed-ups.
    """

    def __init__(
            self,
            fov: float,
            # vision_range_in_cm: int,
            map_size_in_cm: int,
            resolution_in_cm: int,
            height_bins: Sequence[float],
            device: torch.device = torch.device("cpu"),
    ):
        # assert vision_range_in_cm % resolution_in_cm == 0

        self.fov = fov
        # self.vision_range_in_map_units = vision_range_in_cm // resolution_in_cm
        self.map_size_in_cm = map_size_in_cm
        self.resolution_in_cm = resolution_in_cm
        self.height_bins = height_bins
        self.device = device

        self.binned_point_cloud_map = np.zeros(
            (
                self.map_size_in_cm // self.resolution_in_cm,
                self.map_size_in_cm // self.resolution_in_cm,
                len(self.height_bins) + 1,
            ),
            dtype=np.float32,
        )



        self.min_xyz: Optional[np.ndarray] = None

    def update(
            self,
            depth_frame: np.ndarray,
            camera_xyz: np.ndarray,
            camera_rotation: float,
            camera_horizon: float,
    ) -> Dict[str, np.ndarray]:
        """Updates the map with the input depth frame from the agent.

        See the `allenact.embodiedai.mapping.mapping_utils.point_cloud_utils.project_point_cloud_to_map`
        function for more information input parameter definitions. **We assume that the input
        `depth_frame` has depths recorded in meters**.

        # Returns
        Let `map_size = self.map_size_in_cm // self.resolution_in_cm`. Returns a dictionary with keys-values:

        * `"egocentric_update"` - A tensor of shape
            `(vision_range_in_map_units)x(vision_range_in_map_units)x(len(self.height_bins) + 1)` corresponding
            to the binned pointcloud after having been centered on the agent and rotated so that
            points ahead of the agent correspond to larger row indices and points further to the right of the agent
            correspond to larger column indices. Note that by "centered" we mean that one can picture
             the agent as being positioned at (0, vision_range_in_map_units/2) and facing downward. Each entry in this tensor
             is a count equaling the number of points in the pointcloud that, once binned, fell into this
            entry. This is likely the output you want to use if you want to build a model to predict free space from an image.
        * `"allocentric_update"` - A `(map_size)x(map_size)x(len(self.height_bins) + 1)` corresponding
            to `"egocentric_update"` but rotated to the world-space coordinates. This `allocentric_update`
             is what is used to update the internally stored representation of the map.
        *  `"map"` -  A `(map_size)x(map_size)x(len(self.height_bins) + 1)` tensor corresponding
            to the sum of all `"allocentric_update"` values since the last `reset()`.
        ```
        """
        with torch.no_grad():
            assert self.min_xyz is not None, "Please call `reset` before `update`."

            camera_xyz = (
                torch.from_numpy(camera_xyz - self.min_xyz).float().to(self.device)
            )

            current_agent_location = torch.zeros(depth_frame.shape).to(self.device)
            current_agent_location[:] = np.NaN
            current_agent_location[112,112] = 0
            current_agent_world_space_point_cloud = depth_frame_to_world_space_xyz(depth_frame=current_agent_location,camera_world_xyz=camera_xyz,rotation=camera_rotation,horizon=camera_horizon,fov=self.fov,)
            current_agent_binned_indices = project_point_cloud_to_map(xyz_points=current_agent_world_space_point_cloud,bin_axis="y",bins=self.height_bins,map_size=self.binned_point_cloud_map.shape[0],resolution_in_cm=self.resolution_in_cm,flip_row_col=True,).nonzero().squeeze(0)

            depth_frame = torch.from_numpy(depth_frame).to(self.device)
            # depth_frame[
            #     depth_frame
            #     > self.vision_range_in_map_units * self.resolution_in_cm / 100
            #     ] = np.NaN

            depth_frame[depth_frame == -1] = np.NaN


            world_space_point_cloud = depth_frame_to_world_space_xyz(
                depth_frame=depth_frame,
                camera_world_xyz=camera_xyz,
                rotation=camera_rotation,
                horizon=camera_horizon,
                fov=self.fov,
            )

            world_binned_map_update = project_point_cloud_to_map(
                xyz_points=world_space_point_cloud,
                bin_axis="y",
                bins=self.height_bins,
                map_size=self.binned_point_cloud_map.shape[0],
                resolution_in_cm=self.resolution_in_cm,
                flip_row_col=True,
            )
            # Center the cloud on the agent

            agent_centric_point_cloud = rotate_points_to_agent(world_space_point_cloud, self.device, camera_xyz, camera_rotation, self.map_size_in_cm)
            allocentric_update_numpy = world_binned_map_update.cpu().numpy()
            self.binned_point_cloud_map = (
                    self.binned_point_cloud_map + allocentric_update_numpy
            )

            agent_centric_binned_map = project_point_cloud_to_map(
                xyz_points=agent_centric_point_cloud,
                bin_axis="y",
                bins=self.height_bins,
                map_size=self.binned_point_cloud_map.shape[0],
                resolution_in_cm=self.resolution_in_cm,
                flip_row_col=True,
            )
            #TODO what was this?
            # vr = self.vision_range_in_map_units
            # vr_div_2 = self.vision_range_in_map_units // 2
            # width_div_2 = agent_centric_binned_map.shape[1] // 2
            # agent_centric_binned_map = agent_centric_binned_map[
            #                            :vr, (width_div_2 - vr_div_2) : (width_div_2 + vr_div_2), :
            #                            ]





            # things_to_show = world_space_point_cloud; things_to_show = torch.nn.functional.interpolate(things_to_show.permute(2, 0, 1).unsqueeze(0), (40,40)).squeeze(0).permute(1,2,0); show_3d(things_to_show)
            copied_map = self.binned_point_cloud_map.copy()
            x, y, z = current_agent_binned_indices
            copied_map[x, y, z] = -1
            # if depth_frame[~depth_frame.isnan()].sum() > 0:
                # non_nan_items = world_space_point_cloud[~depth_frame.isnan()]
                # self.three_d_points = torch.cat([self.three_d_points, non_nan_items], dim=0)
                # just_object_of_interest_agent_centric = rotate_points_to_agent(self.three_d_points, self.device, camera_xyz, camera_rotation, self.map_size_in_cm) # remove?


                # things_to_show = self.three_d_points;  show_3d(things_to_show)
                # things_to_show = just_object_of_interest_agent_centric;  show_3d(things_to_show)
                # things_to_show = agent_centric_binned_map;  show_3d(things_to_show)
                # things_to_show = agent_centric_binned_map.nonzero();  show_3d(things_to_show)
                # things_to_show = self.all_previous_ones.nonzero();  show_3d(things_to_show)
                # things_to_show = torch.Tensor(self.binned_point_cloud_map).nonzero();  show_3d(things_to_show, map_size = self.binned_point_cloud_map.shape)
                # things_to_show = (agent_centric_binned_map).nonzero();  show_3d(things_to_show, map_size = agent_centric_binned_map.shape)


                # converted_to_agent = rotate_points_to_agent(self.three_d_points, self.device, camera_xyz, camera_rotation, self.map_size_in_cm); converted_bin = project_point_cloud_to_map(xyz_points=converted_to_agent,bin_axis="y",bins=self.height_bins,map_size=self.binned_point_cloud_map.shape[0],resolution_in_cm=self.resolution_in_cm,flip_row_col=True,); things_to_show = (converted_bin).nonzero();  show_3d(things_to_show, map_size = converted_bin.shape)

                # print(self.three_d_points.float().mean(dim=0))
                # print(converted_bin.nonzero().float().mean(dim=0))
                # converted_to_agent = rotate_points_to_agent(agent_centric_point_cloud[(agent_centric_point_cloud == agent_centric_point_cloud)[:,:,0]], self.device, camera_xyz, camera_rotation, self.map_size_in_cm).squeeze(0); print(converted_to_agent.mean(dim=0))

                # converted_bin = project_point_cloud_to_map(xyz_points=self.three_d_points,bin_axis="y",bins=self.height_bins,map_size=self.binned_point_cloud_map.shape[0],resolution_in_cm=self.resolution_in_cm,flip_row_col=True,);things_to_show = (converted_bin).nonzero();  show_3d(things_to_show, map_size = converted_bin.shape)

                # things_to_show = torch.Tensor(copied_map).nonzero();  show_3d(things_to_show, map_size = copied_map.shape)
                # print(current_agent_binned_indices)
                # ForkedPdb().set_trace()

            return {
                "egocentric_update": agent_centric_binned_map.cpu().numpy(),
                "allocentric_update": allocentric_update_numpy,
                "map": self.binned_point_cloud_map,
                'map_with_agent': copied_map,
            }

    def reset(self, min_xyz: np.ndarray):
        """Reset the map.

        Resets the internally stored map.

        # Parameters
        min_xyz : An array of size (3,) corresponding to the minimum possible x, y, and z values that will be observed
            as a point in a pointcloud when calling `.update(...)`. The (world-space) maps returned by calls to `update`
            will have been normalized so the (0,0,:) entry corresponds to these minimum values.
        """
        self.min_xyz = min_xyz
        self.binned_point_cloud_map = np.zeros_like(self.binned_point_cloud_map)
        # self.three_d_points = torch.zeros((0,3))

class KianaReachableBoundsTHORSensor(Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]):
    def __init__(self, margin: float, uuid: str = "scene_bounds", **kwargs: Any):
        observation_space = gym.spaces.Dict(
            {
                "x_range": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf], dtype=np.float32),
                    high=np.array([np.inf, np.inf], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "z_range": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf], dtype=np.float32),
                    high=np.array([np.inf, np.inf], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                # "y_range": gym.spaces.Box(
                #     low=np.array([-np.inf, -np.inf], dtype=np.float32),
                #     high=np.array([np.inf, np.inf], dtype=np.float32),
                #     shape=(2,),
                #     dtype=np.float32,
                # ),
            }
        )
        super().__init__(**prepare_locals_for_super(locals()))

        self.margin = margin
        self._bounds_cache = {}

    @staticmethod
    def get_bounds(
            controller: ai2thor.controller.Controller, margin: float,
    ) -> Dict[str, np.ndarray]:
        event = controller.step("GetReachablePositions")
        positions = event.metadata["actionReturn"]
        min_x = min(p["x"] for p in positions)
        max_x = max(p["x"] for p in positions)
        min_z = min(p["z"] for p in positions)
        max_z = max(p["z"] for p in positions)
        # objects = [x['position'] for x in controller.last_event.metadata["objects"]]
        # min_x = min(o['x'] for o in objects)
        # max_x = max(o['x'] for o in objects)
        # min_y = min(o['y'] for o in objects)
        # max_y = max(o['y'] for o in objects)
        # min_z = min(o['z'] for o in objects)
        # max_z = max(o['z'] for o in objects)

        return {
            "x_range": np.array([min_x - margin, max_x + margin]),
            "z_range": np.array([min_z - margin, max_z + margin]),
            # "x_range": np.array([min_x, max_x]),
            # "z_range": np.array([min_z, max_z]),
            # "y_range": np.array([min_y, max_y]),
        }
    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[Task[RoboThorEnvironment]],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        scene_name = env.controller.last_event.metadata["sceneName"]
        if scene_name not in self._bounds_cache:
            self._bounds_cache[scene_name] = self.get_bounds(
                controller=env.controller, margin=self.margin
            )

        return copy.deepcopy(self._bounds_cache[scene_name])
