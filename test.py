
from dataclasses import dataclass
import os

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl

import lib.utils as utils
from lib.utils import standard_normal_logprob
from lib.utils import build_model_tabular_suhan

import warp as wp
import warp.sim
import warp.sim.render
from lib.warp.urdf_loader import parse_urdf

from robot import Robot
from datasets import KinematicsDataset
from learner import Learner

import copy

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
    num_samples = 100
    num_references = 100
    seed = 100

np.random.seed(args.seed)
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
    q = np.array([-1.3401983 ,  0.64112365,  0.5398765 , -0.7059776,   1.3258765 ,  1.959088, -0.75554395])
    fk = r.get_forward_kinematics(q)
    print(fk)

if __name__ == '__main__':

    run()
    