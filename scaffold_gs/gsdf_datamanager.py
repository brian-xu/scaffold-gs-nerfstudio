"""
GSDF DataManager

Due to the dual architecture requiring both raybundles and cameras, this datamanager returns both.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class GSDFDataManagerConfig(VanillaDataManagerConfig):
    """GSDF DataManager Config"""

    _target: Type = field(default_factory=lambda: GSDFDataManager)
    train_num_images_to_sample_from: int = 1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = 0
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = 1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = 1
    """How many steps to pretrain the Scaffold-GS model before dual training."""
    scaffold_gs_pretrain: int = 20_000


class GSDFDataManager(VanillaDataManager):
    """GSDF DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GSDFDataManagerConfig

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        camera = self.train_dataset.cameras[
            image_batch["image_idx"][0] : image_batch["image_idx"][0] + 1
        ]
        if step == 0 or step > self.config.scaffold_gs_pretrain:
            batch = self.train_pixel_sampler.sample(image_batch)
            ray_indices = batch["indices"]
            ray_bundle = self.train_ray_generator(ray_indices)
            ray_bundle.extra = {}
            ray_bundle.extra["camera"] = camera
            ray_bundle.extra["indices"] = ray_indices
            batch["full_image"] = image_batch["image"][0]
            return ray_bundle, batch
        else:
            image_batch["full_image"] = image_batch["image"][0]
            return camera, image_batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        camera = self.eval_dataset.cameras[
            image_batch["image_idx"][0] : image_batch["image_idx"][0] + 1
        ]
        if step == 0 or step > self.config.scaffold_gs_pretrain:
            batch = self.eval_pixel_sampler.sample(image_batch)
            ray_indices = batch["indices"]
            ray_bundle = self.eval_ray_generator(ray_indices)
            ray_bundle.extra = {}
            ray_bundle.extra["camera"] = camera
            ray_bundle.extra["indices"] = ray_indices
            batch["full_image"] = image_batch["image"][0]
            return ray_bundle, batch
        else:
            image_batch["full_image"] = image_batch["image"][0]
            return camera, image_batch
