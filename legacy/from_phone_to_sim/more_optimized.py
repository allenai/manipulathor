import open3d as o3d
import numpy as np

from manipulathor_utils.debugger_util import ForkedPdb
from utils.from_phone_to_sim.thor_frames_to_pointcloud import save_pointcloud_to_file
from utils.from_phone_to_sim.world_utils import make_k_from_params


def get_point_cloud(color_frames, depth_frames, metadatas):
    h = metadatas[0]['screenHeight']
    w = metadatas[0]['screenWidth']
    fv_deg = metadatas[0]['fov']
    intrinsic_mat = make_k_from_params(h, w, fv_deg)[0]
    fx = intrinsic_mat[0,0]
    fy = intrinsic_mat[1,1]
    x = intrinsic_mat[0, 1]
    x0 = intrinsic_mat[0,2]
    y0 = intrinsic_mat[0,2]
    camera_intrin = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=x0, cy=y0)
    # camera_intrin = o3d.camera.PinholeCameraIntrinsic(    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    result = []
    for i in range(len(color_frames)):
        color = color_frames[0]
        depth = depth_frames[0]
        grayscale = color.mean(axis=-1)
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(grayscale, depth)
        depth_as_image = o3d.geometry.Image((depth).astype(np.uint8))
        rgb = o3d.geometry.Image((color).astype(np.uint8))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth_as_image)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_intrin)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd], zoom=0.5)
        #TODO convert with agent pose as well
        result.append(pcd)
    images = [np.asarray(pcd.colors) for pcd in result]
    depths = [np.asarray(pcd.points) for pcd in result]
    images = np.concatenate(images, axis=0)
    depths = np.concatenate(depths, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(depths)
    pcd.colors = o3d.utility.Vector3dVector(images)
    # ForkedPdb().set_trace()
    # save_pointcloud_to_file(pcd, 'something.ply')

# import matplotlib.pyplot as plt
# plt.subplot(1, 2, 1)
# plt.title('Redwood grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('Redwood depth image')
# plt.imshow(rgbd_image.depth)
# plt.savefig('something.png')