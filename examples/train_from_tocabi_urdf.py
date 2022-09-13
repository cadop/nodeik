from dataclasses import dataclass

import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot, RobotDual
from nodeik.training.datasets import KinematicsDataset
from nodeik.training.learner import Learner

@dataclass
class args:
    layer_type = 'concatsquash'
    dims = '1024-1024-1024-1024'
    num_blocks = 1 
    time_length = 0.5
    train_T = False
    divergence_fn = 'approximate'
    nonlinearity = 'tanh'
    solver = 'dopri5'
    atol = 1e-5
    rtol = 1e-5
    use_gpu = False
    gpu = 0
    rademacher = False
    adjoint = True
    max_epoch = 1000000000

if args.use_gpu:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

wp.init()

def get_robot(filepath):
    r = RobotDual(robot_path=filepath,
                  ee1_link_name='L_Wrist2_Link', 
                  ee2_link_name='R_Wrist2_Link', 
                  joint_map=[12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])

    batch_size = 512
    batch_in_epoch = 500
    val_size = 512
    dataset = KinematicsDataset(r, len_batch=batch_size*batch_in_epoch)
    val_dataset = KinematicsDataset(r, len_batch=val_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size)
    
    return r, dataloader, val_dataloader

def run():
    # URDF file path
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots','tocabi_description', 'dyros_tocabi.urdf')
    
    # Get robot object
    r, dataloader, val_dataloader = get_robot(filepath)

    # Build a CNF model
    model = build_model(args, r.active_joint_dim, condition_dims=14).to(device) # , condition_dims=14 because dual target
    params = sum(p.numel() for p in model.parameters())
    print('parameters', params)

    # Create a learner
    learn = Learner(model, robot=r, std=1.0, num_samples=250, state_dim=r.active_joint_dim, condition_dim=14)
    learn.model_wrapper.device = device
    print('device', device)

    # Training
    trainer = pl.Trainer(max_epochs=args.max_epoch,
                         accelerator='gpu' if device != 'cpu' else None, 
                         gpus=[args.gpu] if device != 'cpu' else None, 
                         check_val_every_n_epoch=1, 
                         log_every_n_steps=10, 
                         default_root_dir=os.path.join(os.path.dirname(__file__), 'checkpoints'))
    trainer.fit(learn,dataloader,val_dataloader)


if __name__ == '__main__':

    run()
    