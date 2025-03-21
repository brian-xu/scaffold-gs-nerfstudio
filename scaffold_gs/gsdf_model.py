from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Shaped
from pytorch_msssim import SSIM
from scaffold_gs.gaussian_splatting.cameras import depth_double_to_normal
from scaffold_gs.scaffold_gs_renderer import prefilter_voxel, scaffold_gs_render
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_max
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.lib_bilagrid import color_correct
from nerfstudio.model_components.losses import MiDaSMSELoss, monosdf_normal_loss
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def rays_from_positions(positions: Shaped[Tensor, "*bs 3"]) -> Shaped[Tensor, "*bs 1"]:
    """Returns ray samples casting to an exact position.

    Args:
        positions: the origin of the samples/frustums
    """
    return RaySamples(
        camera_indices=torch.zeros(1).int().to(device=positions.device),
        frustums=Frustums(
            origins=positions,
            directions=torch.ones_like(positions),
            starts=torch.zeros_like(positions[..., :1]),
            ends=torch.zeros_like(positions[..., :1]),
            pixel_area=torch.ones_like(positions[..., :1]),
        ),
    )


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones(
        (1, 1, d, d), dtype=torch.float32, device=image.device
    )
    return (
        tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d)
        .squeeze(1)
        .permute(1, 2, 0)
    )


@dataclass
class GSDFModelConfig(NeuSFactoModelConfig):
    """GSDF Model Config"""

    _target: Type = field(default_factory=lambda: GSDFModel)
    feat_dim: int = 32
    n_offsets: int = 10
    voxel_size: float = 0.001
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4
    use_feat_bank: bool = False
    appearance_dim: int = 0
    ratio: int = 1
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False
    # for anchor densification
    start_stat: int = 500
    update_from: int = 1500
    update_interval: int = 100
    update_until: int = 30_000
    warmup_length: int = 500
    # densification config
    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002
    scaffold_gs_pretrain: int = 20_000
    normal_loss_warmup: int = 5000

    regularization_from_iter = 15_000
    lambda_depth_normal = 0.05

    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""


class GSDFModel(NeuSFactoModel):
    """Nerfstudio implementation of GSDF

    Args:
        config: GSDF configuration to instantiate model
    """

    config: GSDFModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        if self.seed_points is not None and not self.config.random_init:
            anchor = torch.nn.Parameter(self.seed_points[0])
        else:
            anchor = torch.nn.Parameter(
                (torch.rand((self.config.num_random, 3)) - 0.5)
                * self.config.random_scale
            )
        offset = torch.zeros((anchor.shape[0], self.config.n_offsets, 3)).float().cuda()
        anchor_feat = (
            torch.zeros((anchor.shape[0], self.config.feat_dim)).float().cuda()
        )
        distances, _ = k_nearest_sklearn(anchor.data, 3)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scaling = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 6)))
        rotation = torch.zeros((anchor.shape[0], 4)).cuda()
        rotation[:, 0] = 1
        opacity = inverse_sigmoid(0.1 * torch.ones((anchor.shape[0], 1)).float().cuda())

        opacity_dist_dim = 1 if self.config.add_opacity_dist else 0
        self.mlp_opacity = torch.nn.Sequential(
            MLP(self.config.feat_dim + 3 + opacity_dist_dim, 1, self.config.feat_dim),
            torch.nn.ReLU(True),
            MLP(self.config.feat_dim, 1, self.config.n_offsets),
            torch.nn.Tanh(),
        )
        self.mlp_opacity.train()

        cov_dist_dim = 1 if self.config.add_cov_dist else 0
        self.mlp_cov = torch.nn.Sequential(
            MLP(self.config.feat_dim + 3 + cov_dist_dim, 1, self.config.feat_dim),
            torch.nn.ReLU(True),
            MLP(self.config.feat_dim, 1, 7 * self.config.n_offsets),
        )
        self.mlp_cov.train()

        color_dist_dim = 1 if self.config.add_color_dist else 0
        self.mlp_color = torch.nn.Sequential(
            MLP(
                self.config.feat_dim + 3 + color_dist_dim + self.config.appearance_dim,
                1,
                self.config.feat_dim,
            ),
            torch.nn.ReLU(True),
            MLP(self.config.feat_dim, 1, 3 * self.config.n_offsets),
            torch.nn.Sigmoid(),
        )
        self.mlp_color.train()

        self.mlp_feature_bank = torch.nn.Sequential(
            MLP(3 + 1, 1, self.config.feat_dim),
            torch.nn.ReLU(True),
            MLP(self.config.feat_dim, 1, 3),
            torch.nn.Softmax(dim=1),
        )
        self.mlp_feature_bank.train()

        self.embedding_appearance = Embedding(
            self.num_train_data, self.config.appearance_dim
        )
        self.embedding_appearance.train()

        self.opacity_accum = torch.zeros((anchor.shape[0], 1)).float().cuda()

        self.offset_gradient_accum = (
            torch.zeros((anchor.shape[0] * self.config.n_offsets, 1)).float().cuda()
        )
        self.offset_denom = (
            torch.zeros((anchor.shape[0] * self.config.n_offsets, 1)).float().cuda()
        )

        self.anchor_denom = torch.zeros((anchor.shape[0], 1)).float().cuda()

        self.viewspace_point_tensor = None
        self.neural_opacity = None
        self.visibility_filter = None
        self.offset_selection_mask = None
        self.voxel_visible_mask = None

        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.gauss_params = torch.nn.ParameterDict(
            {
                "anchor": anchor,
                "offset": offset,
                "anchor_feat": anchor_feat,
                "scaling": scaling,
                "rotation": rotation,
                "opacity": opacity,
            }
        )

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.ray_collider, self.collider = self.collider, None

        self.depth_loss = MiDaSMSELoss()

    @property
    def num_points(self):
        return self.anchor.shape[0]

    @property
    def appearance(self):
        return self.embedding_appearance

    @property
    def scaling(self):
        return 1.0 * torch.exp(self.gauss_params["scaling"])

    @property
    def featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def opacity_mlp(self):
        return self.mlp_opacity

    @property
    def cov_mlp(self):
        return self.mlp_cov

    @property
    def color_mlp(self):
        return self.mlp_color

    @property
    def rotation(self):
        return self.gauss_params["rotation"]

    @property
    def anchor(self):
        return self.gauss_params["anchor"]

    @property
    def offset(self):
        return self.gauss_params["offset"]

    @property
    def anchor_feat(self):
        return self.gauss_params["anchor_feat"]

    @property
    def opacity(self):
        return self.gauss_params["opacity"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 45000
        if "anchor" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in [
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "opacity",
            ]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.anchor"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(
                torch.zeros(new_shape, device=self.device)
            )
        super().load_state_dict(dict, **kwargs)

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        if self.step < self.config.update_until and self.step > self.config.start_stat:
            self.update_anchor_adjustment_stats()
        if (
            step < self.config.update_until
            and step > self.config.update_from
            and step % self.config.update_interval == 0
        ):
            if step > self.config.scaffold_gs_pretrain + self.config.warmup_length:
                scaling = self.scaling[:, :3]
                scaling_repeat = (
                    scaling.unsqueeze(dim=1)
                    .repeat([1, self.config.n_offsets, 1])
                    .view([-1, 3])
                )
                gs_positions = (
                    self.anchor.unsqueeze(dim=1)
                    .repeat([1, self.config.n_offsets, 1])
                    .view([-1, 3])
                    + self.offset.view([-1, 3]) * scaling_repeat
                )

                min_point = torch.tensor(
                    self.scene_box.aabb[0],
                    device=gs_positions.device,
                )
                max_point = torch.tensor(
                    self.scene_box.aabb[1],
                    device=gs_positions.device,
                )
                inside_box = (gs_positions > min_point) & (gs_positions < max_point)
                inside_box = inside_box.all(dim=1)

                inside_positions = gs_positions[inside_box]
                # set the sdf of 3D gaussians in the background to 100000.
                xyz_sdf = (
                    torch.ones(gs_positions.shape[0]).to(gs_positions.device) * 100000
                )
                # calculate the sdf of 3D Gaussians in the frontground.
                inside_xyz_sdf = self.field.get_sdf(
                    rays_from_positions(inside_positions)
                ).detach()

                xyz_sdf[inside_box] = inside_xyz_sdf[:, 0]
                # calculate the sdf of anchor points in the frontground
                anchor_positions = self.anchor
                anchor_inside_box = (anchor_positions > min_point) & (
                    anchor_positions < max_point
                )
                anchor_inside_box = anchor_inside_box.all(dim=1)
                anchor_sdf = self.field.get_sdf(
                    rays_from_positions(anchor_positions)
                ).detach()[:, 0]
            else:
                xyz_sdf = None
                anchor_sdf = None
                inside_box = None
                anchor_inside_box = None
            self.adjust_anchor(
                check_interval=self.config.update_interval,
                success_threshold=self.config.success_threshold,
                grad_threshold=self.config.densify_grad_threshold,
                min_opacity=self.config.min_opacity,
                xyz_sdf=xyz_sdf,
                anchor_sdf=anchor_sdf,
                inside_box=inside_box,
                anchor_inside_box=anchor_inside_box,
            )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []

        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step: int):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(
                    (step - self.config.scaffold_gs_pretrain) / N, 0, 1
                )

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            cbs.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            cbs.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min(
                    [
                        1.0,
                        max(
                            0,
                            (step - self.config.scaffold_gs_pretrain) / self.anneal_end,
                        ),
                    ]
                )
                self.field.set_cos_anneal_ratio(anneal)

            cbs.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    func=set_anneal,
                    update_every_num_iters=1,
                )
            )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def get_gaussian_param_groups(self, param_groups) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        for name in [
            "anchor",
            "offset",
            "anchor_feat",
            "scaling",
            "rotation",
            "opacity",
        ]:
            param_groups[name] = [self.gauss_params[name]]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        param_groups = super().get_param_groups()
        self.get_gaussian_param_groups(param_groups=param_groups)
        param_groups["mlp_opacity"] = list(self.opacity_mlp.parameters())
        param_groups["mlp_feature_bank"] = list(self.featurebank_mlp.parameters())
        param_groups["mlp_cov"] = list(self.mlp_cov.parameters())
        param_groups["mlp_color"] = list(self.mlp_color.parameters())
        param_groups["embedding_appearance"] = list(self.appearance.parameters())
        return param_groups

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (
                    self.config.num_downscales
                    - self.step // self.config.resolution_schedule
                ),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(
        width: int, height: int, background: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "background": background,
        }

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera)
        return outs  # type: ignore

    def get_outputs(
        self, ray_bundle: RayBundle
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if isinstance(ray_bundle, RayBundle):
            ray_bundle = self.ray_collider(ray_bundle)
            if not hasattr(ray_bundle, "extra"):
                return super().get_outputs(ray_bundle)
            camera = ray_bundle.extra["camera"]
        elif isinstance(ray_bundle, Cameras):
            camera = ray_bundle
        else:
            print("Called get_outputs with something other than a raybundle or camera")
            return {}

        # during eval the camera gets clone for each raybundle for some reason
        if camera.shape[0] > 1:
            camera = camera[:1]

        assert camera.shape[0] == 1, "Only one camera at a time"
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        background = self._get_background_color()
        voxel_visible_mask = prefilter_voxel(camera, self, background)
        retain_grad = self.step < self.config.update_until and self.step >= 0
        render_pkg = scaffold_gs_render(
            camera,
            self,
            background,
            visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
        )

        (
            render,
            viewspace_point_tensor,
            visibility_filter,
            offset_selection_mask,
            radii,
            scaling,
            opacity,
            gs_depth,
            gs_normal,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["selection_mask"],
            render_pkg["radii"],
            render_pkg["scaling"],
            render_pkg["neural_opacity"],
            render_pkg["expected_depth"],
            render_pkg["gs_normal"],
        )

        rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
        rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
        depth_middepth_normal = depth_double_to_normal(
            camera, rendered_expected_depth, rendered_median_depth
        )
        normal_error_map = 1 - (gs_normal.unsqueeze(0) * depth_middepth_normal).sum(
            dim=1
        )

        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        self.viewspace_point_tensor = viewspace_point_tensor
        self.neural_opacity = opacity
        self.visibility_filter = visibility_filter
        self.offset_selection_mask = offset_selection_mask
        self.voxel_visible_mask = voxel_visible_mask

        rgb = rearrange(render, "c h w -> h w c")
        rgb = torch.clamp(rgb, 0.0, 1.0)

        depth = rearrange(gs_depth, "c h w -> h w c")[..., 0]
        accumulation = depth
        normal = (rearrange(gs_normal, "c h w -> h w c") + 1) / 2

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return_dict = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth.unsqueeze(-1),  # type: ignore
            "normal": normal.squeeze(0),  # type: ignore
            "accumulation": accumulation.unsqueeze(-1),  # type: ignore
            "background": background,  # type: ignore
            "scaling": scaling,  # type: ignore
            "normal_error_map": normal_error_map.squeeze(0),  # type: ignore
        }  # type: ignore

        if self.step > self.config.scaffold_gs_pretrain and isinstance(
            ray_bundle, RayBundle
        ):

            ray_x_indices = ray_bundle.extra["indices"][:, 2] // camera_scale_fac
            ray_x_indices = torch.clamp(ray_x_indices, min=0, max=W - 1).int()
            ray_y_indices = ray_bundle.extra["indices"][:, 1] // camera_scale_fac
            ray_y_indices = torch.clamp(ray_y_indices, min=0, max=H - 1).int()

            gs_depth = depth[ray_y_indices, ray_x_indices]
            gs_normal = normal[ray_y_indices, ray_x_indices]

            neus_outputs = super().get_outputs(ray_bundle)

            if self.training:
                return_dict.update({"eik_grad": neus_outputs["eik_grad"]})

            return_dict["neus_info"] = (
                neus_outputs["rgb"],
                neus_outputs["depth"],
                gs_depth,
                neus_outputs["normal"],
                gs_normal,
                neus_outputs["accumulation"],
            )

        return return_dict

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["full_image"]), outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        if self.training and self.step > self.config.scaffold_gs_pretrain:
            # training statics
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = (
                1.0 / self.field.deviation_network.get_variance().item()
            )

        return metrics_dict

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["full_image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1
            + self.config.ssim_lambda * simloss,
        }

        if self.step > self.config.regularization_from_iter:
            depth_ratio = 0.6

            loss_dict["depth_normal_loss"] = (
                (1 - depth_ratio) * outputs["normal_error_map"][0].mean()
                + depth_ratio * outputs["normal_error_map"][1].mean()
            ) * self.config.lambda_depth_normal

        if self.step > self.config.scaffold_gs_pretrain and self.training:
            scaling_reg = outputs["scaling"].prod(dim=1).mean()
            loss_dict["main_loss"] += 0.01 * scaling_reg

            (
                neus_rgb,
                neus_depth,
                gs_depth,
                neus_normal,
                gs_normal,
                neus_accumulation,
            ) = outputs["neus_info"]

            mask = torch.ones_like(gs_depth).reshape(1, 32, -1).bool()

            loss_weight = (
                1 / 10
                if self.step
                > self.config.scaffold_gs_pretrain + self.config.warmup_length + 15000
                else 1
            )

            normal_loss_mult = (
                0
                if self.step
                < self.config.scaffold_gs_pretrain + self.config.normal_loss_warmup
                else 1
            )

            pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=neus_rgb,
                pred_accumulation=neus_accumulation,
                gt_image=batch["image"].to(self.device),
            )
            loss_dict["neus_loss"] = self.rgb_loss(image, pred_image)

            loss_dict["neus_depth"] = (
                self.depth_loss(
                    neus_depth.reshape(1, 32, -1),
                    gs_depth.detach().reshape(1, 32, -1),
                    mask,
                )
                * self.config.mono_depth_loss_mult
                * loss_weight
            )

            loss_dict["neus_normal"] = (
                monosdf_normal_loss(neus_normal, gs_normal.detach())
                * self.config.mono_normal_loss_mult
                * loss_weight
                * normal_loss_mult
            )

            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = (
                (grad_theta.norm(2, dim=-1) - 1) ** 2
            ).mean() * self.config.eikonal_loss_mult

            if self.step > self.config.scaffold_gs_pretrain + self.config.warmup_length:
                loss_dict["gsdf_loss"] = (
                    self.depth_loss(
                        neus_depth.detach().reshape(1, 32, -1),
                        gs_depth.reshape(1, 32, -1),
                        mask,
                    )
                    * self.config.mono_depth_loss_mult
                    * loss_weight
                    + monosdf_normal_loss(neus_normal.detach(), gs_normal)
                    * self.config.mono_normal_loss_mult
                    * loss_weight
                    * normal_loss_mult
                )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        if "full_image" not in batch:
            gt_img = batch["image"]
        else:
            gt_img = batch["full_image"]
        gt_rgb = self.composite_with_background(
            self.get_gt_img(gt_img), outputs["background"]
        )
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict

    def update_anchor_adjustment_stats(self):
        # update opacity stats
        temp_opacity = self.neural_opacity.clone().view(-1).detach().cuda()
        temp_opacity[temp_opacity < 0] = 0

        temp_opacity = temp_opacity.view([-1, self.config.n_offsets])
        self.opacity_accum[self.voxel_visible_mask] += temp_opacity.sum(
            dim=1, keepdim=True
        )

        # update anchor visiting stats
        self.anchor_denom[self.voxel_visible_mask] += 1

        # update neural gaussian stats
        self.voxel_visible_mask = (
            self.voxel_visible_mask.unsqueeze(dim=1)
            .repeat([1, self.config.n_offsets])
            .view(-1)
        )
        combined_mask = torch.zeros_like(
            self.offset_gradient_accum, dtype=torch.bool
        ).squeeze(dim=1)
        combined_mask[self.voxel_visible_mask] = self.offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = self.visibility_filter

        grad_norm = torch.norm(
            self.viewspace_point_tensor.grad[self.visibility_filter, :2],
            dim=-1,
            keepdim=True,
        )
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for optim_name in self.optimizers:
            if optim_name == name:
                optim = self.optimizers[optim_name]
                group = optim.param_groups[0]
                stored_state = optim.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optim.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                optim.state[group["params"][0]] = stored_state

                optimizable_tensors[optim_name] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for optim_name in self.optimizers:
            if (
                "mlp" in optim_name
                or "conv" in optim_name
                or "feat_base" in optim_name
                or "embedding" in optim_name
                or "field" in optim_name
                or "proposal" in optim_name
            ):
                continue
            optim = self.optimizers[optim_name]
            group = optim.param_groups[0]
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[optim_name]
            stored_state = optim.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del optim.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optim.state[group["params"][0]] = stored_state

            else:
                group["params"][0] = torch.nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
            optimizable_tensors[optim_name] = group["params"][0]

        return optimizable_tensors

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for optim_name in self.optimizers:
            if (
                "mlp" in optim_name
                or "conv" in optim_name
                or "feat_base" in optim_name
                or "embedding" in optim_name
                or "field" in optim_name
                or "proposal" in optim_name
            ):
                continue

            optim = self.optimizers[optim_name]
            group = optim.param_groups[0]

            stored_state = optim.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optim.state[group["params"][0]]

                new_param = group["params"][0][mask]
                if optim_name == "scaling":
                    temp = new_param[:, 3:]
                    temp[temp > 0.05] = 0.05
                    new_param[:, 3:] = temp
                group["params"][0] = torch.nn.Parameter(
                    (new_param.requires_grad_(True))
                )
                optim.state[group["params"][0]] = stored_state
            else:
                new_param = group["params"][0][mask]
                if optim_name == "scaling":
                    temp = new_param[:, 3:]
                    temp[temp > 0.05] = 0.05
                    new_param[:, 3:] = temp
                group["params"][0] = torch.nn.Parameter(
                    (new_param.requires_grad_(True))
                )
            optimizable_tensors[optim_name] = group["params"][0]

        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self.gauss_params["anchor"] = optimizable_tensors["anchor"]
        self.gauss_params["offset"] = optimizable_tensors["offset"]
        self.gauss_params["anchor_feat"] = optimizable_tensors["anchor_feat"]
        self.gauss_params["opacity"] = optimizable_tensors["opacity"]
        self.gauss_params["scaling"] = optimizable_tensors["scaling"]
        self.gauss_params["rotation"] = optimizable_tensors["rotation"]

    def anchor_growing(self, grads, threshold, offset_mask):
        ##
        init_length = self.anchor.shape[0] * self.config.n_offsets
        for i in range(self.config.update_depth):
            # update threshold
            cur_threshold = threshold * ((self.config.update_hierachy_factor // 2) ** i)
            # mask from grad threshold
            candidate_mask = grads >= cur_threshold
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            # random pick
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.anchor.shape[0] * self.config.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat(
                    [
                        candidate_mask,
                        torch.zeros(length_inc).bool().cuda(),
                    ],
                    dim=0,
                )

            all_xyz = self.anchor.unsqueeze(dim=1) + self.offset * self.scaling[
                :, :3
            ].unsqueeze(dim=1)

            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.config.update_init_factor // (
                self.config.update_hierachy_factor**i
            )
            cur_size = self.config.voxel_size * size_factor

            grid_coords = torch.round(self.anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0
            )

            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (
                    1 if grid_coords.shape[0] % chunk_size != 0 else 0
                )
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (
                        (
                            selected_grid_coords_unique.unsqueeze(1)
                            == grid_coords[i * chunk_size : (i + 1) * chunk_size, :]
                        )
                        .all(-1)
                        .any(-1)
                        .view(-1)
                    )
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (
                    (selected_grid_coords_unique.unsqueeze(1) == grid_coords)
                    .all(-1)
                    .any(-1)
                    .view(-1)
                )

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = (
                    torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda()
                    * cur_size
                )  # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = (
                    torch.zeros([candidate_anchor.shape[0], 4]).float().cuda()
                )
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1)).float().cuda()
                )

                new_feat = (
                    self.anchor_feat.unsqueeze(dim=1)
                    .repeat([1, self.config.n_offsets, 1])
                    .view([-1, self.config.feat_dim])[candidate_mask]
                )

                new_feat = scatter_max(
                    new_feat,
                    inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)),
                    dim=0,
                )[0][remove_duplicates]

                new_offsets = (
                    torch.zeros_like(candidate_anchor)
                    .unsqueeze(dim=1)
                    .repeat([1, self.config.n_offsets, 1])
                    .float()
                    .cuda()
                )

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_denom = torch.cat(
                    [
                        self.anchor_denom,
                        torch.zeros([new_opacities.shape[0], 1]).float().cuda(),
                    ],
                    dim=0,
                )
                del self.anchor_denom
                self.anchor_denom = temp_anchor_denom

                temp_opacity_accum = torch.cat(
                    [
                        self.opacity_accum,
                        torch.zeros([new_opacities.shape[0], 1]).float().cuda(),
                    ],
                    dim=0,
                )
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self.gauss_params["anchor"] = optimizable_tensors["anchor"]
                self.gauss_params["offset"] = optimizable_tensors["offset"]
                self.gauss_params["anchor_feat"] = optimizable_tensors["anchor_feat"]
                self.gauss_params["opacity"] = optimizable_tensors["opacity"]
                self.gauss_params["scaling"] = optimizable_tensors["scaling"]
                self.gauss_params["rotation"] = optimizable_tensors["rotation"]

    def adjust_anchor(
        self,
        add_contents=None,
        check_interval=100,
        success_threshold=0.8,
        grad_threshold=0.0002,
        min_opacity=0.005,
        xyz_sdf=None,
        anchor_sdf=None,
        inside_box=None,
        anchor_inside_box=None,
        growing_weight=0.0002,
    ):

        if xyz_sdf is not None:
            # Activate function (Gaussian) for sdf.
            def simple_sdf_activate(x, sigma=0.01):
                return torch.exp(-(x**2) / sigma)

        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k, 1]
        grads[grads.isnan()] = 0.0

        grads_norm = torch.norm(grads, dim=-1)

        if xyz_sdf is not None:
            xyz_sdf_activated = simple_sdf_activate(xyz_sdf)
            xyz_sdf_activated[~inside_box] = 0.0
            grow_alpha = growing_weight  # 0.0002
            weight_prune = 1
            # update the grads_norm according to the sdf value
            grads_norm = grads_norm + grow_alpha * xyz_sdf_activated

        offset_mask = (
            self.offset_denom > check_interval * success_threshold * 0.5
        ).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros(
            [
                self.anchor.shape[0] * self.config.n_offsets
                - self.offset_denom.shape[0],
                1,
            ],
            dtype=torch.int32,
            device=self.offset_denom.device,
        )
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [
                self.anchor.shape[0] * self.config.n_offsets
                - self.offset_gradient_accum.shape[0],
                1,
            ],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device,
        )
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum, padding_offset_gradient_accum], dim=0
        )

        anchor_opacity_sdf_accum = self.opacity_accum

        if anchor_sdf is not None:

            anchor_sdf_activated = simple_sdf_activate(anchor_sdf)
            anchor_sdf_activated[~anchor_inside_box] = 1
            padding_length = self.anchor.shape[0] - anchor_sdf_activated.shape[0]
            padding_ones = torch.ones([padding_length]).to(self.anchor.device)
            padded_anchor_sdf_activated = torch.cat(
                [anchor_sdf_activated, padding_ones], dim=0
            )
            # update the opacity_accum with the anchor sdf value.
            anchor_opacity_sdf_accum = (
                self.opacity_accum
                - weight_prune
                * self.anchor_denom
                * (1 - padded_anchor_sdf_activated.unsqueeze(dim=1))
            )

        # # prune anchors
        prune_mask = (
            anchor_opacity_sdf_accum < min_opacity * self.anchor_denom
        ).squeeze(dim=1)
        anchors_mask = (self.anchor_denom > check_interval * success_threshold).squeeze(
            dim=1
        )  # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        extent = self.scene_box.get_diagonal_length() * 0.55
        scaling_mask = self.scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_and(prune_mask, scaling_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.config.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view(
            [-1, self.config.n_offsets]
        )[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = (
                torch.zeros([anchors_mask.sum(), 1]).float().cuda()
            )
            self.anchor_denom[anchors_mask] = (
                torch.zeros([anchors_mask.sum(), 1]).float().cuda()
            )

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_denom = self.anchor_denom[~prune_mask]
        del self.anchor_denom
        self.anchor_denom = temp_anchor_denom

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)
