"""
GSDF Depth Sampler.

Experimental and obsoleted code.
"""

# from typing import Optional

# import torch
# from jaxtyping import Float
# from torch import Tensor

# from nerfstudio.cameras.rays import RayBundle, RaySamples
# from nerfstudio.model_components.ray_samplers import Sampler


# class GSDFDepthSampler(Sampler):
#     """Sample based on GS depth values

#     Args:
#         num_samples: Number of samples per ray
#         train_stratified: Randomize location within each bin during training.
#     """

#     def __init__(
#         self,
#         num_samples: Optional[int] = 128,
#         train_stratified: bool = True,
#         single_jitter: bool = False,
#         radius: float = 3.0,
#     ) -> None:
#         super().__init__(num_samples=num_samples)
#         self.train_stratified = train_stratified
#         self.single_jitter = single_jitter
#         self.radius = radius

#     def generate_ray_samples(
#         self,
#         ray_bundle: Optional[RayBundle] = None,
#         gs_depth: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
#         sdf_depth: Optional[Float[Tensor, "*batch num_samples 1"]] = None,
#     ) -> RaySamples:
#         """Generates position samples given depth values.

#         Args:
#             ray_bundle: Rays to generate samples for
#             gs_depth: Depth values from the GS branch
#             sdf_depth: Depth values from the SDF branch

#         Returns:
#             Positions and deltas for samples along a ray
#         """

#         if ray_bundle is None or gs_depth is None or sdf_depth is None:
#             raise ValueError("ray_samples, gs_depth, and sdf_depth must be provided")

#         num_samples = self.num_samples
#         assert num_samples is not None
#         num_rays = ray_bundle.origins.shape[0]

#         bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[
#             None, ...
#         ]  # [1, num_samples+1]

#         if self.train_stratified and self.training:
#             if self.single_jitter:
#                 t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
#             else:
#                 t_rand = torch.rand(
#                     (num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device
#                 )
#             bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
#             bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
#             bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
#             bins = bin_lower + (bin_upper - bin_lower) * t_rand
#         s_near = gs_depth - self.radius * sdf_depth.abs()
#         s_far = gs_depth + self.radius * sdf_depth.abs()

#         def spacing_to_euclidean_fn(x):
#             return x * s_far + (1 - x) * s_near

#         euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]

#         ray_samples = ray_bundle.get_ray_samples(
#             bin_starts=euclidean_bins[..., :-1, None],
#             bin_ends=euclidean_bins[..., 1:, None],
#             spacing_starts=bins[..., :-1, None],
#             spacing_ends=bins[..., 1:, None],
#             spacing_to_euclidean_fn=spacing_to_euclidean_fn,
#         )

#         return ray_samples
