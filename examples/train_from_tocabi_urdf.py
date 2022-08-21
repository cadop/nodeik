from dataclasses import dataclass

import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot
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
    gpu = 0
    rademacher = False
    adjoint = True

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

wp.init()

def get_robot(filepath):

    r = Robot(robot_path=filepath,end_joint_index=-1, ee_link_index=12)

    val_size = 256
    dataset = KinematicsDataset(r, len_batch=4096*500)
    val_dataset = KinematicsDataset(r, len_batch=val_size)
    dataloader = DataLoader(dataset, batch_size=4096)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size)
    
    return r, dataloader, val_dataloader

def run():
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots','tocabi_description', 'dyros_tocabi_gansi.urdf')
    r, dataloader, val_dataloader = get_robot(filepath)
    model = build_model(args, r.joint_dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    print('parameters', params)
    learn = Learner(model, robot=r, std=1.0, num_samples=4)
    learn.model_wrapper.device = device

    trainer = pl.Trainer(max_epochs=1000000000,
                         accelerator='gpu', 
                         gpus=[args.gpu], 
                         check_val_every_n_epoch=1, 
                         log_every_n_steps=10, 
                         default_root_dir=os.path.join(os.path.dirname(__file__), 'checkpoints'))
    trainer.fit(learn,dataloader,val_dataloader)


if __name__ == '__main__':

    run()
    