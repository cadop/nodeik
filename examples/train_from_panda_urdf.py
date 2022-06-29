from dataclasses import dataclass

import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot
from nodeik.training.datasets import KinematicsDataset
from nodeik.training.learner import Learner


@dataclass
class args:
    layer_type = 'concatsquash'
    dims = '64-64-64'
    num_blocks = 1 
    time_length = 0.5
    train_T = False
    divergence_fn = 'brute_force'
    nonlinearity = 'tanh'
    solver = 'dopri5'
    atol = 1e-5
    rtol = 1e-5
    gpu = 0
    rademacher = False

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

wp.init()

def get_robot(filepath):

    r = Robot(robot_path=filepath)
    
    dataset = KinematicsDataset(r)
    dataloader = DataLoader(dataset, batch_size=4096)
    
    return r, dataloader


def run():
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots','franka_panda', 'panda_arm.urdf')

    r, dataloader = get_robot(filepath)
    model = build_model(args, 7).to(device)
    learn = Learner(model, dataloader)

    trainer = pl.Trainer(max_epochs=1000000000,accelerator='gpu', devices=1)
    trainer.fit(learn)


if __name__ == '__main__':

    run()
    