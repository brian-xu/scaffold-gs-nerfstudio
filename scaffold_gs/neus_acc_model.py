# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Type

import nerfacc
import torch
from scaffold_gs.neus_acc_sampler import NeuSAccSampler
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig


@dataclass
class NeuSAccModelConfig(NeuSModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: NeuSAccModel)
    sky_loss_mult: float = 0.01
    """Sky segmentation normal consistency loss multiplier."""


class NeuSAccModel(NeuSModel):
    """VolSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: NeuSAccModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # voxel surface bybrid sampler from NeuralReconW
        self.sampler = NeuSAccSampler(
            aabb=self.scene_box.aabb, neus_sampler=self.sampler
        )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # add sampler call backs
        sdf_fn = lambda x: self.field.forward_geonetwork(x)[:, 0].contiguous()
        inv_s = self.field.deviation_network.get_variance
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_binary_grid,
                kwargs={"sdf_fn": sdf_fn, "inv_s": inv_s},
            )
        )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_step_size,
                kwargs={"inv_s": inv_s},
            )
        )

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        # bootstrap with original Neus
        if self.sampler._update_counter.item() <= 0:
            return super().get_outputs(ray_bundle)

        ray_samples, ray_indices = self.sampler(
            ray_bundle, sdf_fn=self.field.get_sdf, alpha_fn=self.field.get_alpha
        )

        if ray_samples.shape[0] > 0:
            field_outputs = self.field(ray_samples, return_alphas=True)

            n_rays = ray_bundle.shape[0]
            weights, trans = nerfacc.render_weight_from_alpha(
                field_outputs[FieldHeadNames.ALPHA].flatten(),
                ray_indices=ray_indices,
                n_rays=n_rays,
            )
            rgb = nerfacc.accumulate_along_rays(
                weights,
                values=field_outputs[FieldHeadNames.RGB],
                ray_indices=ray_indices,
                n_rays=n_rays,
            )
            normal = nerfacc.accumulate_along_rays(
                weights,
                values=field_outputs[FieldHeadNames.NORMALS],
                ray_indices=ray_indices,
                n_rays=n_rays,
            )

            accumulation = nerfacc.accumulate_along_rays(
                weights, values=None, ray_indices=ray_indices, n_rays=n_rays
            )
            depth = nerfacc.accumulate_along_rays(
                weights,
                values=(ray_samples.frustums.starts + ray_samples.frustums.ends) / 2,
                ray_indices=ray_indices,
                n_rays=n_rays,
            )

            # the rendered depth is point-to-point distance and we should convert to depth
            depth = depth / ray_bundle.metadata["directions_norm"]

            # background model
            if self.config.background_model != "none":
                assert isinstance(
                    self.field_background, torch.nn.Module
                ), "field_background should be a module"
                assert ray_bundle.fars is not None, "fars is required in ray_bundle"
                # sample inversely from far to 1000 and points and forward the bg model
                ray_bundle.nears = ray_bundle.fars
                assert ray_bundle.fars is not None
                ray_bundle.fars = (
                    torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg
                )

                ray_samples_bg = self.sampler_bg(ray_bundle)
                # use the same background model for both density field and occupancy field
                assert not isinstance(self.field_background, Parameter)
                field_outputs_bg = self.field_background(ray_samples_bg)
                weights_bg = ray_samples_bg.get_weights(
                    field_outputs_bg[FieldHeadNames.DENSITY]
                )

                rgb_bg = self.renderer_rgb(
                    rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg
                )
                depth_bg = self.renderer_depth(
                    weights=weights_bg, ray_samples=ray_samples_bg
                )
                accumulation_bg = self.renderer_accumulation(weights=weights_bg)

                bg_transmittance = torch.ones(
                    [
                        n_rays,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                bg_transmittance.index_reduce_(
                    0,
                    index=ray_indices,
                    source=(1 - field_outputs[FieldHeadNames.ALPHA].flatten()),
                    reduce="prod",
                    include_self=False,
                )

                bg_transmittance = bg_transmittance.unsqueeze(-1).repeat(1, 3)

                # merge background color to foregound color
                rgb = rgb + bg_transmittance * rgb_bg

                bg_outputs = {
                    "bg_rgb": rgb_bg,
                    "bg_accumulation": accumulation_bg,
                    "bg_depth": depth_bg,
                    "bg_weights": weights_bg,
                }
            else:
                bg_outputs = {}

            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "normal": normal,
            }
            outputs.update(bg_outputs)

            if self.training:
                grad_points = field_outputs[FieldHeadNames.GRADIENT]
                outputs.update({"eik_grad": grad_points})
        else:
            zeros = torch.zeros(
                (ray_bundle.shape[0], 3), dtype=torch.float32, device=self.device
            )
            outputs = {
                "rgb": zeros,
                "accumulation": zeros[:, :1],
                "depth": zeros[:, :1],
                "normal": zeros,
            }
            if self.training:
                outputs.update({"eik_grad": zeros})

        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics = super().get_metrics_dict(outputs, batch)
        metrics["acc_step_size"] = self.sampler.step_size
        return metrics
