# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
Unofficial implementation of [Scaffold-GS](https://github.com/city-super/Scaffold-GS) as a [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) extension.

https://github.com/user-attachments/assets/f7194455-e4a0-4c60-9608-837ef635f22b

## GSDF: 3DGS Meets SDF for Improved Neural Rendering and Reconstruction
Now includes an implementation of [GSDF](https://github.com/city-super/GSDF) as a [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) extension.

#### Note: there are several differences when compared to the reference implementation, as we build off of the nerfstudio implementation of NeuS.

## Status

- [ ] Functionality
  - [x] Correctly rendering and training
  - [x] Anchor adjustment
  - [x] GSDF implementation
  - [ ] Debugging
- [ ] Visualization
  - [x] Fix viser functionality
  - [ ] ...
- [ ] Extras
  - [ ] Scaffold-GS baked splat export
  - [ ] GSDF mesh export
  - [ ] ...


## Environment

This repository has been tested on the following environment. Other similar configurations should also work.

```
pop-os 22.04
cuda 11.8
gcc 10.5
python 3.10.16
numpy 1.26.4
torch 2.1.2+cu118
torchvision 0.16.2+cu118
```

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd scaffold-gs-nerfstudio/
pip install -e .
ns-install-cli
```

You must also install the correct torch_scatter for your environment ([https://pytorch-geometric.com/whl/torch-2.1.2%2Bcu118.html](https://pytorch-geometric.com/whl/torch-2.1.2%2Bcu118.html))

## Running Scaffold-GS
This repository creates a new Nerfstudio method named "scaffold-gs". To train with it, run the command:
```
ns-train scaffold-gs --data [PATH]
```

GSDF can be run via the following command:
```
# for indoor scenes
ns-train gsdf --data [PATH] --pipeline.model.sdf-field.inside-outside True
# for object scenes
ns-train gsdf --data [PATH] --pipeline.model.sdf-field.inside-outside False
```

Unbounded scenes are not supported, but might work with `--pipeline.model.sdf-field.inside-outside True`.

## Acknowledgements

Thanks to the authors of [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for the reference implementation, which we build upon heavily.

### Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

```bibtex
@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20654--20664},
  year={2024}
}
```

Thanks to the authors of [GSDF](https://github.com/city-super/GSDF)  for the reference implementation of GSDF, which we build upon heavily.

### GSDF: 3DGS Meets SDF for Improved Neural Rendering and Reconstruction

```bibtex
@article{yu2024gsdf,
  title={Gsdf: 3dgs meets sdf for improved rendering and reconstruction},
  author={Yu, Mulin and Lu, Tao and Xu, Linning and Jiang, Lihan and Xiangli, Yuanbo and Dai, Bo},
  journal={arXiv preprint arXiv:2403.16964},
  year={2024}
}
```

Thanks to the authors of [Splatfacto-360](https://github.com/myuito3/splatfacto-360) for their code converting nerfstudio cameras to the COLMAP format.

Thanks to the authors of [dn-splatter](https://github.com/maturk/dn-splatter) for providing examples of how to export splats from a nerfstudio extension.