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

## How to train
The example training code can be found in `examples` directory.
```sh
cd examples
python train_panda_urdf.py
```

## How to visualize the reuslt
TBA

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
