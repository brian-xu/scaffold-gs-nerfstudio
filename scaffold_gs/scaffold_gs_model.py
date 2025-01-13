from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from einops import rearrange
from pytorch_msssim import SSIM
from scaffold_gs.scaffold_gs_renderer import prefilter_voxel, scaffold_gs_render
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
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
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn
from nerfstudio.utils.misc import torch_compile


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


@dataclass
class ScaffoldGSModelConfig(ModelConfig):
    """Scaffold-GS Model Config"""

    _target: Type = field(default_factory=lambda: ScaffoldGSModel)
    feat_dim: int = 32
    n_offsets: int = 5
    voxel_size: float = 0.01
    update_depth: int = 3
    update_init_factor: int = 100
    update_hierachy_factor: int = 4
    use_feat_bank: bool = False
    appearance_dim: int = 32
    ratio: int = 1
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False
    # for anchor densification
    update_from: int = 1500
    update_interval: int = 100
    update_until: int = 15_000
    warmup_length: int = 500
    # densification config
    min_opacity = 0.005
    success_threshold = 0.8
    densify_grad_threshold = 0.0002
    """period of steps where gaussians are culled and densified"""
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
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="off")
    )
    """Config of the camera optimizer to use"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""


class ScaffoldGSModel(Model):
    """Nerfstudio implementation of Scaffold-GS

    Args:
        config: Scaffold-GS configuration to instantiate model
    """

    config: ScaffoldGSModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):

        def inverse_sigmoid(x):
            return torch.log(x / (1 - x))

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
        rotation = torch.zeros((anchor.shape[0], 4), device="cuda")
        rotation[:, 0] = 1
        opacity = inverse_sigmoid(
            0.1 * torch.ones((anchor.shape[0], 1), dtype=torch.float, device="cuda")
        )

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

        self.opacity_accum = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

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

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

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
        self.step = 30000
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
        newp = dict["gauss_params.means"].shape[0]
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
        # TODO: implement anchor adjustment
        # if (
        #     step < self.config.update_until
        #     and step > self.config.update_from
        #     and step % self.config.update_interval == 0
        # ):
        #     self.adjust_anchor(
        #         check_interval=self.config.update_interval,
        #         success_threshold=self.config.success_threshold,
        #         grad_threshold=self.config.densify_grad_threshold,
        #         min_opacity=self.config.min_opacity,
        #     )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
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

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in [
                "anchor",
                "offset",
                "anchor_feat",
                "scaling",
                "rotation",
                "opacity",
            ]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        gps["mlp_opacity"] = list(self.opacity_mlp.parameters())
        gps["mlp_feature_bank"] = list(self.featurebank_mlp.parameters())
        gps["mlp_cov"] = list(self.mlp_cov.parameters())
        gps["mlp_color"] = list(self.mlp_color.parameters())
        gps["embedding_appearance"] = list(self.appearance.parameters())
        return gps

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

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

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

        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        (
            render,
            viewspace_point_tensor,
            visibility_filter,
            offset_selection_mask,
            radii,
            scaling,
            opacity,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["selection_mask"],
            render_pkg["radii"],
            render_pkg["scaling"],
            render_pkg["neural_opacity"],
        )

        rgb = rearrange(render, "c h w -> h w c")
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

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
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
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
            self.get_gt_img(batch["image"]), outputs["background"]
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

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

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
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

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
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
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
