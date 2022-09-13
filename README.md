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
