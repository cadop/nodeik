# NODE IK: Neural ODE Inverse Kinematics


## Requirements
- python >= 3.7
- pytorch 1.0.1
- matplotlib
- sklearn
- torchdiffeq
- urdfpy
- warp-lang
- usd-core
- pytorch_lightning
- pyquaternion
- tqdm

### Version Numbers
As of 2023, some breaking changes have happened. 

Install:
- `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
- `numpy==1.22.3`
- `pytorch_lightning==1.5.10`
- `warp-lang==0.2.3`

### Git LFS
We use git lfs to reduce the `dae` file size on clone. Please install git lfs before cloning the repo. Otherwise install after and use git lfs fetch. 

## How to train
The example training code can be found in `examples` directory.
```sh
cd examples
python train_panda_urdf.py
```

## How to visualize the reuslt
The example visualization code can be found in `examples` directory.

```sh
cd examples
python visualize_panda_urdf.py
```

This script creates `usd` files. [NVIDA Omniverse](https://developer.nvidia.com/nvidia-omniverse-platform) can open these files and you can see the generated visual results.

## How to evaluate the trained model
The example evaluation code can be found in `examples` directory.

```sh
cd examples
python evaluation_panda_urdf.py
```

The model checkpoint can be designated by modifying `args`.

## References
`nodeik/layers` are orignated from `FFJORD` and `SoftFlow` repos. Thank the authors for these codes.
- FFJORD: https://github.com/rtqichen/ffjord
- SoftFlow: https://github.com/ANLGBOY/SoftFlow
