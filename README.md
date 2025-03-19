# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
Unofficial implementation of [Scaffold-GS](https://github.com/city-super/Scaffold-GS) as a [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) extension.

https://github.com/user-attachments/assets/f7194455-e4a0-4c60-9608-837ef635f22b

Additionally, we include depth+normal optimization from [RaDe-GS](https://github.com/BaowenZ/RaDe-GS).

https://github.com/user-attachments/assets/65e1e584-f0d8-441f-b432-9c28c4620b21


## GSDF: 3DGS Meets SDF for Improved Neural Rendering and Reconstruction
Now includes an implementation of [GSDF](https://github.com/city-super/GSDF) as a [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) extension.

#### Note: there are several differences when compared to the reference implementation, as we build off of the nerfstudio implementation of NeuS.

## Neus-Acc: NeuS with empty space skipping based on nerfacc
Now includes a port of NeuS-Acc from [SDFStudio](https://github.com/autonomousvision/sdfstudio), ported for recent versions of nerfstudio and gsplat.

#### Note: Neus-Acc implementation is currently experimental.

## Status

- [ ] Functionality
  - [x] Correctly rendering and training
  - [x] Anchor adjustment
  - [ ] Debugging
- [ ] Visualization
  - [x] Fix viser functionality
  - [ ] ...
- [ ] Extras
  - [x] GSDF implementation
  - [ ] GSDF mesh export
  - [ ] NeuS-Acc port
  - [ ] Scaffold-GS mesh export
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
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/quickstart/installation.html). Clone or fork this repository and run the commands:

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

NeuS-Acc can be run via the following command:
```
ns-train neus-acc --data [PATH]
```


## Acknowledgements

Thanks to the authors of [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for the reference implementation, which we build upon heavily.

Thanks to the authors of [GSDF](https://github.com/city-super/GSDF) for the reference implementation of GSDF, which we build upon heavily.

Thanks to the authors of [SDFStudio](https://github.com/autonomousvision/sdfstudio) for their implementation of NeuS-acc.

Thanks to the authors of [Splatfacto-360](https://github.com/myuito3/splatfacto-360) for their code converting nerfstudio cameras to the COLMAP format.

Thanks to the authors of [RaDe-GS](https://github.com/BaowenZ/RaDe-GS) for their differentiable depth + normal rasterization code.

Thanks to the authors of [dn-splatter](https://github.com/maturk/dn-splatter) for providing examples of how to export splats from a nerfstudio extension.
