# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
Unofficial implementation of [Scaffold-GS](https://github.com/city-super/Scaffold-GS) as a [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) extension.

https://github.com/user-attachments/assets/f7194455-e4a0-4c60-9608-837ef635f22b

## Status

- [ ] Functionality
  - [x] Correctly rendering and training
  - [x] Anchor adjustment
  - [ ] ...
- [ ] Visualization
  - [x] Fix viser functionality
  - [ ] ...
- [ ] Extras
  - [ ] Mesh export
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

Thanks to the authors of [Splatfacto-360](https://github.com/myuito3/splatfacto-360) for their code converting nerfstudio cameras to the COLMAP format.
