# import open3d as o3d
import torch

import utils.from_phone_to_sim.world_utils as wu

def get_camera_params(fov, h, w):
    xy = wu.pixel_grid(h, w)
    K = wu.make_k_from_params(h, w, fov)
    rays = wu.camera_rays(xy, K)
    return xy, K, rays

def points_from_rgbd(xy, K, rays, meta, frame, depth_frame):
    z = wu.depth_to_z([depth_frame])
    uxyz = wu.depth_to_clip(z, xy=xy)
    cxyz = wu.clip_to_camera(uxyz, K=K)

    normals = wu.camera_to_normals(cxyz, z)
    fil = wu.filter_normals(normals, rays) * wu.filter_depth(z)
    fil = fil.view(-1)

    cwxyz = wu.camera_to_world(cxyz, meta=[meta])
    cwxyz = cwxyz.view(-1, 3)[fil]

    cwnormals = wu.camera_to_world(normals.unsqueeze(-1), meta=[meta], no_translate=True)
    cwnormals = cwnormals.view(-1, 3)[fil]

    cwrgb = torch.from_numpy(frame.copy()).transpose(0, 1) / 255.0
    cwrgb = cwrgb.reshape(-1, 3)[fil]

    return cwxyz, cwnormals, cwrgb

def frames_to_world_points(metadatas, frames, depth_frames):
    h = metadatas[0]['screenHeight']
    w = metadatas[0]['screenWidth']
    fov = metadatas[0]['fov']
    xy, K, rays = get_camera_params(fov, h, w)

    xyz, normals, rgb = [], [], []

    for meta, frame, depth_frame in zip(metadatas, frames, depth_frames):
        cwxyz, cwnormals, cwrgb = points_from_rgbd(xy, K, rays, meta, frame, depth_frame)
        xyz.append(cwxyz)
        normals.append(cwnormals)
        rgb.append(cwrgb)

    return xyz, normals, rgb

def world_points_to_pointcloud(xyz, normals, rgb, voxel_size=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(torch.cat(xyz, dim=0).numpy())
    pc.normals = o3d.utility.Vector3dVector(torch.cat(normals, dim=0).numpy())
    pc.colors = o3d.utility.Vector3dVector(torch.cat(rgb, dim=0).numpy())
    if voxel_size is not None:
        pc = pc.voxel_down_sample(voxel_size)
    return pc

def load_pointcloud_from_file(pc_filepath):
    return o3d.io.read_point_cloud(pc_filepath)

def save_pointcloud_to_file(pc, filepath):
    o3d.io.write_point_cloud(filepath, pc, write_ascii=False)

def pointcloud_to_mesh(pc):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=10)
    return mesh

def write_mesh(mesh, filename):
    o3d.io.write_triangle_mesh(filename, mesh, write_ascii=False)