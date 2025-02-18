"""
GSDF DataManager

Due to the dual architecture requiring both raybundles and cameras, this datamanager returns both.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union, cast

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.pixel_samplers import (
    PatchPixelSamplerConfig,
    PixelSampler,
    PixelSamplerConfig,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class GSDFDataManagerConfig(FullImageDatamanagerConfig):
    """GSDF DataManager Config"""

    _target: Type = field(default_factory=lambda: GSDFDataManager)
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    patch_size: int = 1
    """Size of patch to sample from. If > 1, patch-based sampling will be used."""
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    """Specifies the pixel sampler used to sample pixels from images."""
    scaffold_gs_pretrain: int = 100.0
    """How many steps to pretrain the Scaffold-GS model before dual training."""


class GSDFDataManager(FullImageDatamanager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GSDFDataManagerConfig
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: GSDFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

    def _get_pixel_sampler(
        self, dataset: TDataset, num_rays_per_batch: int
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        if (
            self.config.patch_size > 1
            and type(self.config.pixel_sampler) is PixelSamplerConfig
        ):
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (
            dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        ).all()
        if is_equirectangular.any():
            CONSOLE.print(
                "[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler."
            )

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device)
        )
        super().setup_train()

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        self.eval_pixel_sampler = self._get_pixel_sampler(
            self.eval_dataset, self.config.eval_num_rays_per_batch
        )
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device)
        )
        super().setup_eval()

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        # TODO: default return rays
        camera, data = super().next_train(step)
        self.train_count += 1
        if step > self.config.scaffold_gs_pretrain:
            assert self.train_pixel_sampler is not None
            assert isinstance(data, dict)
            data["image"] = data["image"].unsqueeze(0)
            data["image_idx"] = torch.Tensor([data["image_idx"]])
            batch = self.train_pixel_sampler.sample(data)
            ray_indices = batch["indices"]
            ray_bundle = self.train_ray_generator(ray_indices)
            data["image"] = data["image"].squeeze(0)
            camera.metadata["raycast_image"] = batch["image"]
            camera.metadata["indices"] = batch["indices"]
            camera.metadata["ray_bundle"] = ray_bundle
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        camera, data = super().next_eval_image(step)
        self.eval_count += 1
        if step > self.config.scaffold_gs_pretrain:
            assert self.eval_pixel_sampler is not None
            assert isinstance(data, dict)
            data["image"] = data["image"].unsqueeze(0)
            data["image_idx"] = torch.Tensor([data["image_idx"]])
            batch = self.eval_pixel_sampler.sample(data)
            ray_indices = batch["indices"]
            ray_bundle = self.eval_ray_generator(ray_indices)
            data["image"] = data["image"].squeeze(0)
            camera.metadata["raycast_image"] = batch["image"]
            camera.metadata["indices"] = batch["indices"]
            camera.metadata["ray_bundle"] = ray_bundle
        return camera, data
