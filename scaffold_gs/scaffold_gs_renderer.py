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
from einops import repeat
from scaffold_gs.gaussian_splatting.cameras import (
    ColmapCamera,
    convert_to_colmap_camera,
)

import gsplat
from gsplat.cuda._wrapper import fully_fused_projection


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

    colmap_camera = convert_to_colmap_camera(viewpoint_camera)

    # Set up rasterization configuration
    tanfovx = math.tan(colmap_camera.FoVx * 0.5)
    tanfovy = math.tan(colmap_camera.FoVy * 0.5)

    focal_length_x = colmap_camera.image_width / (2 * tanfovx)
    focal_length_y = colmap_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, colmap_camera.image_width / 2.0],
            [0, focal_length_y, colmap_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    viewmat = colmap_camera.world_view_transform.transpose(0, 1)  # [4, 4]
    (
        render_colors,
        render_alphas,
        expected_depths,
        median_depths,
        expected_normals,
        info,
    ) = gsplat.rasterization(
        means=xyz,  # [N, 3]
        quats=rot,  # [N, 4]
        scales=scaling,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=color,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(colmap_camera.image_width),
        height=int(colmap_camera.image_height),
        packed=False,
        sh_degree=None,
        render_mode="RGB",
        return_depth_normal=True,
    )
    
    rendered_image = render_colors[0].permute(2, 0, 1)
    rendered_alpha = render_alphas[0].permute(2, 0, 1)
    rendered_expected_depth = expected_depths[0].permute(2, 0, 1)
    rendered_median_depth = median_depths[0].permute(2, 0, 1)
    rendered_normal = expected_normals[0].permute(2, 0, 1)
    
    radii = info["radii"].squeeze(0)  # [N,]
    try:
        info["means2d"].retain_grad()  # [1, N, 2]
    except:
        pass

    return {
        "render": rendered_image,
        "mask": rendered_alpha,
        "expected_depth": rendered_expected_depth,
        "median_depth": rendered_median_depth,
        "viewspace_points": info["means2d"],
        "visibility_filter": torch.sum(radii, dim=-1).squeeze() > 0,
        "radii": radii,
        "selection_mask": mask,
        "neural_opacity": neural_opacity,
        "scaling": scaling,
        "gs_normal": rendered_normal,
    }


def prefilter_voxel(viewpoint_camera: ColmapCamera, pc):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    colmap_camera = convert_to_colmap_camera(viewpoint_camera)

    means = pc.anchor
    scales = pc.scaling[:, :3]
    quats = pc.rotation
    # Set up rasterization configuration
    tanfovx = math.tan(colmap_camera.FoVx * 0.5)
    tanfovy = math.tan(colmap_camera.FoVy * 0.5)
    focal_length_x = colmap_camera.image_width / (2 * tanfovx)
    focal_length_y = colmap_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor(
        [
            [focal_length_x, 0, colmap_camera.image_width / 2.0],
            [0, focal_length_y, colmap_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )[None]
    viewmats = colmap_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        None,  # covars,
        quats,
        scales,
        viewmats,
        Ks,
        int(colmap_camera.image_width),
        int(colmap_camera.image_height),
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=False,
    )

    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations, ray_ts, ray_planes, normals = (
        proj_results
    )
    camera_ids, gaussian_ids = None, None

    return torch.sum(radii, dim=-1).squeeze() > 0
