from __future__ import annotations

from scaffold_gs.gsdf_datamanager import GSDFDataManagerConfig
from scaffold_gs.gsdf_model import GSDFModelConfig
from scaffold_gs.gsdf_scheduler import DelayedCosineDecaySchedulerConfig
from scaffold_gs.neus_acc_model import NeuSAccModelConfig
from scaffold_gs.scaffold_gs_model import ScaffoldGSModelConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.datasets.sdf_dataset import SDFDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
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
            model=ScaffoldGSModelConfig(),
        ),
        optimizers={
            "anchor": {
                "optimizer": AdamOptimizerConfig(lr=0, eps=1e-15),
                "scheduler": None,
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

gsdf = MethodSpecification(
    config=TrainerConfig(
        method_name="gsdf",
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=500,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=45000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=GSDFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    load_3D_points=True, center_method="focus"
                ),
            ),
            model=GSDFModelConfig(
                sdf_field=SDFFieldConfig(
                    num_layers=2,
                    bias=0.5,
                    beta_init=0.3,
                ),
                mono_depth_loss_mult=0.05,
                mono_normal_loss_mult=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=3e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(
                    max_steps=45001, milestones=(20_000, 21500, 30000, 38000)
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": DelayedCosineDecaySchedulerConfig(
                    pretrain_steps=20_000,
                    warm_up_end=500,
                    learning_rate_alpha=0.05,
                    max_steps=45000,
                ),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": DelayedCosineDecaySchedulerConfig(
                    pretrain_steps=20_000,
                    warm_up_end=500,
                    learning_rate_alpha=0.05,
                    max_steps=45000,
                ),
            },
            "anchor": {
                "optimizer": AdamOptimizerConfig(lr=0, eps=1e-15),
                "scheduler": None,
            },
            "offset": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.01,
                    lr_final=0.0001,
                    max_steps=45000,
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
                    max_steps=45000,
                ),
            },
            "mlp_cov": {
                "optimizer": AdamOptimizerConfig(lr=0.004, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.004,
                    lr_final=0.004,
                    max_steps=45000,
                ),
            },
            "mlp_color": {
                "optimizer": AdamOptimizerConfig(lr=0.008, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.008,
                    lr_final=0.00005,
                    max_steps=45000,
                ),
            },
            "mlp_feature_bank": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.01,
                    lr_final=0.00001,
                    max_steps=45000,
                ),
            },
            "embedding_appearance": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_pre_warmup=0.05,
                    lr_final=0.0005,
                    max_steps=45000,
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=45000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="GSDF method.",
)

neus_acc = MethodSpecification(
    config=TrainerConfig(
        method_name="neus-acc",
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[SDFDataset],
                dataparser=SDFStudioDataParserConfig(),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
            ),
            model=NeuSAccModelConfig(eval_num_rays_per_chunk=1024),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000
                ),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeuS-acc method.",
)
