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
from scaffold_gs.gaussian_splatting.cameras import convert_to_colmap_camera


def generate_neural_gaussians(viewpoint_camera, pc, visible_mask=None):
    if visible_mask is None:
        visible_mask = torch.ones(
            pc.anchor.shape[0], dtype=torch.bool, device=pc.anchor.device
        )

    colmap_camera = convert_to_colmap_camera(viewpoint_camera)
    ## get view properties for anchor

    feat = pc.anchor_feat[visible_mask]
    anchor = pc.anchor[visible_mask]
    grid_offsets = pc.offset[visible_mask]
    grid_scaling = pc.scaling[visible_mask]

    ob_view = anchor - colmap_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.config.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)

        bank_weight = pc.featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

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
    if pc.config.appearance_dim > 0:
        if (
            viewpoint_camera.metadata is not None
            and "cam_idx" in viewpoint_camera.metadata
        ):
            cam_idx = viewpoint_camera.metadata["cam_idx"]
        else:
            cam_idx = 1
        camera_indicies = (
            torch.ones_like(
                cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
            )
            * cam_idx
        )
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        # TODO: interpolate camera indices for novel cameras
        appearance = pc.appearance(camera_indicies)

    # get offset's opacity
    if pc.config.add_opacity_dist:
        neural_opacity = pc.opacity_mlp(cat_local_view)  # [N, k]
    else:
        neural_opacity = pc.opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = neural_opacity > 0.0
    mask = mask.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.config.appearance_dim > 0:
        if pc.config.add_color_dist:
            color = pc.color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.config.add_color_dist:
            color = pc.color_mlp(cat_local_view)
        else:
            color = pc.color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * pc.config.n_offsets, 3])  # [mask]

    # get offset's cov
    if pc.config.add_cov_dist:
        scale_rot = pc.cov_mlp(cat_local_view)
    else:
        scale_rot = pc.cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.config.n_offsets, 7])  # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3])  # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(
        concatenated, "n (c) -> (n k) (c)", k=pc.config.n_offsets
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
    rots = torch.nn.functional.normalize(scale_rot[:, 3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    return xyz, color, opacity, scaling, rots, neural_opacity, mask


def scaffold_gs_render(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    kernel_size=0.3,
    scaling_modifier=1.0,
    visible_mask=None,
    retain_grad=False,
    require_depth=True,
    require_coord=True,
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

    colmap_camera = convert_to_colmap_camera(viewpoint_camera)

    # Set up rasterization configuration
    tanfovx = math.tan(colmap_camera.FoVx * 0.5)
    tanfovy = math.tan(colmap_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(colmap_camera.image_height),
        image_width=int(colmap_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=colmap_camera.world_view_transform,
        projmatrix=colmap_camera.full_proj_transform,
        sh_degree=1,
        campos=colmap_camera.camera_center,
        prefiltered=False,
        require_coord=require_coord,
        require_depth=require_depth,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (
        rendered_image,
        radii,
        rendered_expected_coord,
        rendered_median_coord,
        rendered_expected_depth,
        rendered_median_depth,
        rendered_alpha,
        rendered_normal,
    ) = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,
        "mask": rendered_alpha,
        "expected_coord": rendered_expected_coord,
        "median_coord": rendered_median_coord,
        "expected_depth": rendered_expected_depth,
        "median_depth": rendered_median_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "selection_mask": mask,
        "neural_opacity": neural_opacity,
        "scaling": scaling,
        "gs_normal": rendered_normal,
    }


def prefilter_voxel(
    viewpoint_camera,
    pc,
    bg_color: torch.Tensor,
    kernel_size=0.0,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    colmap_camera = convert_to_colmap_camera(viewpoint_camera)

    # Set up rasterization configuration
    tanfovx = math.tan(colmap_camera.FoVx * 0.5)
    tanfovy = math.tan(colmap_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(colmap_camera.image_height),
        image_width=int(colmap_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=colmap_camera.world_view_transform,
        projmatrix=colmap_camera.full_proj_transform,
        sh_degree=1,
        campos=colmap_camera.camera_center,
        prefiltered=False,
        debug=False,
        require_depth=True,
        require_coord=True,
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
