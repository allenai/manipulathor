import math

import torch
from scipy.spatial.transform import Rotation


def get_fov_h_w(controller):
    fov = controller.last_event.metadata['fov']
    h = controller.last_event.metadata['screenHeight']
    w = controller.last_event.metadata['screenWidth']
    return fov, h, w

def get_rotation_matrices(rotations, horizons):
    # rotation, horizon in radians with shape (N,)
    assert rotations.shape[0] == horizons.shape[0]

    cax = torch.cos(horizons)
    sax = torch.sin(horizons)
    rx = torch.zeros(horizons.shape[0], 3, 3)
    rx[:, 0, 0] = 1
    rx[:, 1, 1] = cax
    rx[:, 1, 2] = -sax
    rx[:, 2, 1] = sax
    rx[:, 2, 2] = cax

    R_euler = Rotation.from_matrix(rx).as_euler('YXZ', degrees=False)
    R_euler[:, 0] = rotations

    R = Rotation.from_euler('YXZ', R_euler, degrees=False).as_matrix()
    R = torch.tensor(R, dtype=torch.float32)

    return R

def camera_at_agent_pose(camera_y, agent_pose):
    '''
    Parameters
      agent_pose: { 'position' : xyz_dict, 'y_rotation' : int (degrees), 'horizon' : int (degrees) }
    Returns
      camera_positions: (x, y, z)
      camera_rotations: (r_x, r_y, r_z)
    '''
    camera_position = (agent_pose['position']['x'], camera_y, agent_pose['position']['z'])

    rotation = torch.deg2rad(torch.tensor([agent_pose['y_rotation']], dtype=float))
    horizon = torch.deg2rad(torch.tensor([agent_pose['horizon']], dtype=float))
    R_mat = get_rotation_matrices(rotation, horizon)[0]
    R = Rotation.from_matrix(R_mat)
    camera_rotation = R.as_euler('YXZ', degrees=True)[[1, 0, 2]]  # yxz -> xyz

    return camera_position, camera_rotation

def vertical_to_horizontal_fov(vertical_fov_in_degrees: float, height: float, width: float):
    assert 0 < vertical_fov_in_degrees < 180
    aspect_ratio = width / height
    vertical_fov_in_rads = (math.pi / 180) * vertical_fov_in_degrees
    return (
            (180 / math.pi)
            * math.atan(math.tan(vertical_fov_in_rads * 0.5) * aspect_ratio)
            * 2
    )

# intrinsic matrix
def make_k_from_params(h, w, fv_deg):
    fh = vertical_to_horizontal_fov(fv_deg, h, w) * math.pi / 180
    fv = fv_deg * math.pi / 180
    k = [
        [w / (2 * torch.tan(torch.tensor(fh / 2))), 0, w / 2],
        [0, -h / (2 * torch.tan(torch.tensor(fv / 2))), h / 2],
        [0, 0, 1],
    ]
    return torch.tensor(k).view(1, 3, 3)  # 1 x 3 x 3

def make_k(controller):
    h = controller.last_event.metadata['screenHeight']
    w = controller.last_event.metadata['screenWidth']
    fov = controller.last_event.metadata['fov']
    return make_k_from_params(h, w, fov)

# extrinsic params
def pose_to_meta(pose):
    return {
        "cameraPosition" : pose["position"],
        "agent" : {
            "rotation" : pose["agent_rotation"],
            "cameraHorizon" : pose["agent_horizon"]
        }
    }

def xyzrh_batch(metadata):
    xyzrh = []
    for meta in metadata:
        xyzrh.append(
            torch.tensor(
                [meta["cameraPosition"][x] for x in "xyz"]
                + [
                    meta["agent"]["rotation"]["y"] * math.pi / 180.0,
                    meta["agent"]["cameraHorizon"] * math.pi / 180.0,
                    ]
            )
        )
    return torch.stack(xyzrh, dim=0)  # b x 5

# extrinsic matrices
def make_pose(xyzrh):
    bsize = xyzrh.shape[0]

    ay = xyzrh[:, -2]
    cay = torch.cos(ay)
    say = torch.sin(ay)
    ry = torch.zeros(bsize, 3, 3)
    ry[:, 0, 0] = cay
    ry[:, 0, 2] = -say
    ry[:, 1, 1] = 1  # flip y
    ry[:, 2, 0] = say
    ry[:, 2, 2] = cay

    ax = xyzrh[:, -1]
    cax = torch.cos(ax)
    sax = torch.sin(ax)
    rx = torch.zeros(bsize, 3, 3)
    rx[:, 0, 0] = 1
    rx[:, 1, 1] = cax
    rx[:, 1, 2] = sax
    rx[:, 2, 1] = -sax
    rx[:, 2, 2] = cax

    R = torch.matmul(rx, ry)  # b x 3 x 3
    # the center of projections has to be 0, 0, 0 in the camera system
    pos = xyzrh[:, :3].unsqueeze(-1)  # b x 3 x 1
    t = -torch.matmul(R, pos)  # b x 3 x 1
    return torch.cat([R, t], dim=-1)  # b x 3 x 4


def pixel_grid(height, width):
    row, col = torch.meshgrid([torch.arange(0.5, height), torch.arange(0.5, width)])
    xy = torch.stack([col, row], dim=-1).transpose(0, 1).view(-1, width, height, 2)
    return xy  # 1 x w x h x 2

def depth_to_z(depths):
    z = []
    for depth_frame in depths:
        z.append(
            torch.from_numpy(depth_frame).transpose(0, 1).unsqueeze(-1)
        )
    z = torch.stack(z, dim=0)  # b x w x h x 1
    return z

def depth_to_clip(z, xy=None):
    if xy is None:
        xy = pixel_grid(z.shape[-2], z.shape[-3])  # 1 x w x h x 2
    xyz = xy * z  # b x w x h x 2
    uxyz = torch.cat([xyz, z], dim=-1).view(z.shape[:3] + (3,))  # b x w x h x 3

    return uxyz


def camera_to_normals(cxyz, z):
    # z is b x w x h x 1
    z = torch.nn.functional.pad(z.squeeze(-1), (1, 1, 1, 1))  # b x w x h

    # cxyz is b x w x h x 3
    cxyz = torch.nn.functional.pad(
        cxyz.squeeze(-1), (0, 0, 1, 1, 1, 1)
    )  # do not pad xyz

    # left
    vx1 = -cxyz[:, 1:-1, 1:-1, :] + cxyz[:, :-2, 1:-1, :]  # b x w x h x 3
    dzx1 = torch.abs(-z[:, 1:-1, 1:-1] + z[:, :-2, 1:-1])
    # right
    vx2 = cxyz[:, 2:, 1:-1, :] - cxyz[:, 1:-1, 1:-1, :]  # b x w x h x 3
    dzx2 = torch.abs(z[:, 2:, 1:-1] - z[:, 1:-1, 1:-1])

    # up
    vy1 = -cxyz[:, 1:-1, 1:-1, :] + cxyz[:, 1:-1, :-2, :]  # b x w x h x 3
    dzy1 = torch.abs(-z[:, 1:-1, 1:-1] + z[:, 1:-1, :-2])
    # down
    vy2 = cxyz[:, 1:-1, 2:, :] - cxyz[:, 1:-1, 1:-1, :]  # b x w x h x 3
    dzy2 = torch.abs(z[:, 1:-1, 2:] - z[:, 1:-1, 1:-1])

    # up_right
    v1 = vx2  # right
    v2 = vy1  # up

    right_down = ((dzx2 < dzx1) * (dzy2 < dzy1)).unsqueeze(-1).expand_as(vy1)
    v1 = torch.where(right_down, vy2, v1)  # down
    v2 = torch.where(right_down, vx2, v2)  # right

    left_up = ((dzx1 < dzx2) * (dzy1 < dzy2)).unsqueeze(-1).expand_as(vy1)
    v1 = torch.where(left_up, vy1, v1)  # up
    v2 = torch.where(left_up, vx1, v2)  # left

    down_left = ((dzx1 < dzx2) * (dzy2 < dzy1)).unsqueeze(-1).expand_as(vy1)
    v1 = torch.where(down_left, vx1, v1)  # left
    v2 = torch.where(down_left, vy2, v2)  # down

    norm = torch.cross(v2, v1, dim=-1)
    return torch.nn.functional.normalize(norm, dim=-1)  # b x w x h x 3


def filter_depth(z, near=0.5, far=2.0):
    z = z.squeeze(-1)  # b x w x h
    return (z > near) * (z < far)


def filter_normals(normals, rays, thres=0.5):
    # normals b x w x h x 3
    # rays 1 x w x h x 3
    normals = normals.unsqueeze(-2)
    rays = rays.unsqueeze(-1)
    return torch.matmul(-normals, rays).squeeze(-1).squeeze(-1) >= thres


def shade_normals(normals, light_direction=None):
    if light_direction is None:
        # light_direction = torch.tensor([0.577, -0.577, 0.577])
        light_direction = torch.tensor([0.0, 0.0, 1.0])

    return 0.5 + 0.5 * torch.matmul(normals, light_direction)


def get_cop(Rt):
    return -torch.matmul(Rt[..., :-1].transpose(-2, -1), Rt[..., -1:])  # b x 3 x 1


def clip_to_world(uxyz, K=None, Rt=None, cs=None):
    if K is None:
        assert cs is not None
        K = make_k(cs[0])
    if Rt is None:
        assert cs is not None
        Rt = make_pose(xyzrh_batch([
            c.last_event.metadata for c in cs
        ]))

    KRinv = torch.inverse(torch.matmul(K, Rt[..., :-1])).view(
        K.shape[0], 1, 1, 3, 3
    )  # b x 3 x 3 -> b x 1 x 1 x 3 x 3
    wxyz = torch.matmul(KRinv, uxyz.unsqueeze(-1)).view(-1, 3) + get_cop(Rt).view(
        Rt.shape[0], 1, 1, 3
    )  # b x w x h x 3
    return wxyz


def clip_to_camera(uxyz, K=None, cs=None):
    if K is None:
        assert cs is not None
        K = make_k(cs[0])

    Kinv = torch.inverse(K).view(
        K.shape[0], 1, 1, 3, 3
    )  # b x 3 x 3 -> b x 1 x 1 x 3 x 3
    cxyz = torch.matmul(Kinv, uxyz.unsqueeze(-1))  # b x w x h x 3
    return cxyz


def camera_rays(xy, K=None, cs=None):
    if K is None:
        assert cs is not None
        K = make_k(cs[0])

    Kinv = torch.inverse(K).view(
        K.shape[0], 1, 1, 3, 3
    )  # b x 3 x 3 -> b x 1 x 1 x 3 x 3

    uxyz = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1).view(
        xy.shape[:3] + (3,)
    )  # 1 x w x h x 3

    cxyz = torch.matmul(Kinv, uxyz.unsqueeze(-1)).squeeze(-1)  # b x w x h x 3
    return torch.nn.functional.normalize(cxyz, dim=-1)  # b x w x h x 3


def camera_to_world(cxyz, Rt=None, meta=None, no_translate=False):
    if Rt is None:
        assert meta is not None
        Rt = make_pose(xyzrh_batch(meta))

    Rinv = torch.transpose(Rt[..., :-1], 1, 2).view(
        Rt.shape[0], 1, 1, 3, 3
    )  # b x 3 x 3 -> b x 1 x 1 x 3 x 3

    if no_translate:
        wxyz = torch.matmul(Rinv, cxyz).squeeze(-1)  # b x w x h x 3
    else:
        wxyz = torch.matmul(Rinv, cxyz).squeeze(-1) + get_cop(Rt).view(
            Rt.shape[0], 1, 1, 3
        )  # b x w x h x 3
    return wxyz


def world_points_to_ply(fname, wxyz, normals=None, rgb=None):
    ply_header_template = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex {npoints}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "{normals_header}{rgb_header}end_header\n"
    )

    normals_header = (
        ""
        if normals is None
        else "property float nx\n" "property float ny\n" "property float nz\n"
    )

    rgb_header = (
        ""
        if rgb is None
        else "property uchar red\n" "property uchar green\n" "property uchar blue\n"
    )

    wxyz = wxyz.view(-1, 3)  # b x w x h 3 -> (b * w * h) x 3
    if normals is not None:
        normals = normals.view(-1, 3)  # b x w x h 3 -> (b * w * h) x 3

    with open(fname, "w") as f:
        f.writelines(
            ply_header_template.format(
                npoints=wxyz.shape[0],
                normals_header=normals_header,
                rgb_header=rgb_header,
            )
        )
        for it, point in enumerate(wxyz):
            if normals is not None:
                normals_str = (
                    f" {normals[it][0]} {normals[it][2]} {normals[it][1]}"  # swap y<->z
                )
            else:
                normals_str = ""
            if rgb is not None:
                rgb_str = f" {rgb[it][0]} {rgb[it][1]} {rgb[it][2]}"
            else:
                rgb_str = ""
            f.write(
                f"{point[0]} {point[2]} {point[1]}{normals_str}{rgb_str}\n"
            )  # swap y<->z


def voxel_grid_from_world_points(wxyz, max_vox_per_dim=200, rel_margin=1.05):
    wxyz = wxyz.view(-1, 3)  # b x w x h 3 -> (b * w * h) x 3

    minx = torch.min(wxyz[:, 0])
    maxx = torch.max(wxyz[:, 0])
    miny = torch.min(wxyz[:, 1])
    maxy = torch.max(wxyz[:, 1])
    minz = torch.min(wxyz[:, 2])
    maxz = torch.max(wxyz[:, 2])

    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    cz = (minz + maxz) / 2

    minx = cx - (cx - minx) * rel_margin
    miny = cy - (cy - miny) * rel_margin
    minz = cz - (cz - minz) * rel_margin
    maxx = cx - (cx - maxx) * rel_margin
    maxy = cy - (cy - maxy) * rel_margin
    maxz = cz - (cz - maxz) * rel_margin

    nvox = max_vox_per_dim
    vox_size = math.ceil(max(maxx - minx, maxy - miny, maxz - minz)) / nvox
    nx = int(math.ceil(maxx - minx) / vox_size)
    ny = int(math.ceil(maxy - miny) / vox_size)
    nz = int(math.ceil(maxz - minz) / vox_size)

    px, py, pz = torch.meshgrid(
        (torch.arange(0, nx) + 0.5) * vox_size + minx,
        (torch.arange(0, ny) + 0.5) * vox_size + miny,
        (torch.arange(0, nz) + 0.5) * vox_size + minz,
        )
    wvoxels = torch.stack([px, py, pz, torch.ones_like(px)], dim=-1).view(
        1, nx, ny, nz, 4
    )

    return wvoxels  # 1 x nx x ny x nz x 4