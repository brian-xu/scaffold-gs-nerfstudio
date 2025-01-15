#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

## Taken from https://github.com/myuito3/splatfacto-360.

import math

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras as NerfstudioCamera


class ColmapCamera:
    def __init__(
        self,
        R,
        T,
        FoVx,
        FoVy,
        image_width,
        image_height,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def convert_to_colmap_camera(camera: NerfstudioCamera):
    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = torch.eye(4).to(camera.camera_to_worlds)
    c2w[:3, :] = camera.camera_to_worlds[0]
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3].T  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())
    fovx = focal2fov(K[0, 0, 0], W)
    fovy = focal2fov(K[0, 1, 1], H)

    return ColmapCamera(R=R, T=T, FoVx=fovx, FoVy=fovy, image_height=H, image_width=W)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
