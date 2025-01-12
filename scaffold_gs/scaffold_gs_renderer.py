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
import math

import torch
from diff_scaffold_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import repeat

from nerfstudio.utils.misc import torch_compile


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


def getProjectionMatrix(fovX, fovY, znear=0.01, zfar=100):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device="cuda")

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def generate_neural_gaussians(camera, model, visible_mask=None):
    if visible_mask is None:
        visible_mask = torch.ones(
            model.anchor.shape[0], dtype=torch.bool, device=model.anchor.device
        )
    ## get view properties for anchor
    world_view_transform = get_viewmat(camera.camera_to_worlds)[0].T
    camera_center = world_view_transform.inverse()[3, :3]

    feat = model.anchor_feat[visible_mask]
    anchor = model.anchor[visible_mask]
    grid_offsets = model.offset[visible_mask]
    grid_scaling = model.scaling[visible_mask]

    ob_view = anchor - camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if model.config.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)

        bank_weight = model.featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = (
            feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
            + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
            + feat[:, ::1, :1] * bank_weight[:, :, 2:]
        )
        feat = feat.squeeze(dim=-1)  # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]p
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
    if (
        model.config.appearance_dim > 0
        and camera.metadata is not None
        and "cam_idx" in camera.metadata
    ):
        camera_indicies = (
            torch.ones_like(
                cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
            )
            * camera.metadata["cam_idx"]
        )
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = model.appearance(camera_indicies)

    # get offset's opacity
    if model.config.add_opacity_dist:
        neural_opacity = model.opacity_mlp(cat_local_view)  # [N, k]
    else:
        neural_opacity = model.opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = neural_opacity > 0.0
    mask = mask.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    # get offset's color
    if (
        model.config.appearance_dim > 0
        and camera.metadata is not None
        and "cam_idx" in camera.metadata
    ):
        if model.config.add_color_dist:
            color = model.color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = model.color_mlp(
                torch.cat([cat_local_view_wodist, appearance], dim=1)
            )
    else:
        if model.config.add_color_dist:
            color = model.color_mlp(cat_local_view)
        else:
            color = model.color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * model.config.n_offsets, 3])  # [mask]

    # get offset's cov
    if model.config.add_cov_dist:
        scale_rot = model.cov_mlp(cat_local_view)
    else:
        scale_rot = model.cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape(
        [anchor.shape[0] * model.config.n_offsets, 7]
    )  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(
        concatenated, "n (c) -> (n k) (c)", k=model.config.n_offsets
    )
    concatenated_all = torch.cat(
        [concatenated_repeated, color, scale_rot, offsets], dim=-1
    )
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split(
        [6, 3, 3, 7, 3], dim=-1
    )

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3]
    )  # * (1+torch.sigmoid(repeat_dist))
    quats = scale_rot[:, 3:7]

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    return xyz, color, opacity, scaling, quats, neural_opacity, mask


def scaffold_gs_render(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(
        viewpoint_camera, pc, visible_mask
    )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(xyz, dtype=pc.anchor.dtype, requires_grad=True, device="cuda")
        + 0
    )
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fx * 0.5)
    tanfovy = math.tan(viewpoint_camera.fy * 0.5)

    world_view_transform = get_viewmat(
        pc.camera_optimizer.apply_to_camera(viewpoint_camera)
    )[0].T
    projection_matrix = getProjectionMatrix(viewpoint_camera.fx, viewpoint_camera.fy)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=1,
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "selection_mask": mask,
        "neural_opacity": neural_opacity,
        "scaling": scaling,
    }


def prefilter_voxel(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.anchor, dtype=pc.anchor.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fx * 0.5)
    tanfovy = math.tan(viewpoint_camera.fy * 0.5)

    world_view_transform = get_viewmat(
        pc.camera_optimizer.apply_to_camera(viewpoint_camera)
    )[0].T
    projection_matrix = getProjectionMatrix(viewpoint_camera.fx, viewpoint_camera.fy)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=1,
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.scaling
    rotations = pc.rotation

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return radii_pure > 0
