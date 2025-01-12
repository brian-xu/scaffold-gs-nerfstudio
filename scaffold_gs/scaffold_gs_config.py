"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from scaffold_gs.scaffold_gs_model import ScaffoldGSModelConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

scaffold_gs = MethodSpecification(
    config=TrainerConfig(
        method_name="scaffold-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=ScaffoldGSModelConfig(appearance_dim=0),
        ),
        optimizers={
            "anchor": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.01,
                    lr_final=0.0001,
                    max_steps=30000,
                ),
            },
            "offset": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.01,
                    lr_final=0.0001,
                    max_steps=30000,
                ),
            },
            "anchor_feat": {
                "optimizer": AdamOptimizerConfig(lr=0.0075, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.007, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15),
                "scheduler": None,
            },
            "mlp_opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.002,
                    lr_final=0.00002,
                    max_steps=30000,
                ),
            },
            "mlp_opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.002, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.002,
                    lr_final=0.00002,
                    max_steps=30000,
                ),
            },
            "mlp_cov": {
                "optimizer": AdamOptimizerConfig(lr=0.004, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.004,
                    lr_final=0.004,
                    max_steps=30000,
                ),
            },
            "mlp_cov": {
                "optimizer": AdamOptimizerConfig(lr=0.004, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.004,
                    lr_final=0.004,
                    max_steps=30000,
                ),
            },
            "mlp_color": {
                "optimizer": AdamOptimizerConfig(lr=0.008, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.008,
                    lr_final=0.00005,
                    max_steps=30000,
                ),
            },
            "mlp_color": {
                "optimizer": AdamOptimizerConfig(lr=0.008, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.008,
                    lr_final=0.00005,
                    max_steps=30000,
                ),
            },
            "mlp_feature_bank": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.01,
                    lr_final=0.00001,
                    max_steps=30000,
                ),
            },
            "embedding_appearance": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.05,
                    lr_final=0.0005,
                    max_steps=30000,
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="ScaffoldGS method.",
)
