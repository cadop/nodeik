from dataclasses import dataclass
import os
import time
import pickle
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl

import lib.utils as utils
from lib.utils import standard_normal_logprob
from lib.utils import count_nfe, count_total_time
from lib.utils import build_model_tabular_suhan

import warp as wp
import warp.sim
import warp.sim.render
from lib.warp.urdf_loader import parse_urdf


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

class Robot():
    def __init__(self, render=True, num_envs=1, device='cpu', robot_path='panda_arm.urdf',start_joint_index=0, end_joint_index=7):
        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))
        print("It should be nodeik dir")

        # change the current working directory (URDF load trick)
        # os.chdir('{}/assets/robots'.format(cwd)) # Not a good trick because it doesn't work, just load the file relative to script

        parse_urdf(
            robot_path, 
            builder,
            xform=wp.transform(np.array((0, 0.0, 0.0)), wp.quat_from_axis_angle((1.0, 0.0, 0.0), 0.0)),
            floating=False, 
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1)

        # restore cwd
        os.chdir(cwd)
        self.model = builder.finalize(device)
        self.model.ground = False

        self.device = device
        self.state = self.model.state()

        self.joint_dim = len(self.model.joint_q.numpy())
        self.start_joint_index = start_joint_index
        self.end_joint_index = end_joint_index
        self.joint_limit_lb = self.model.joint_limit_lower.numpy()[start_joint_index:end_joint_index]
        self.joint_limit_ub = self.model.joint_limit_upper.numpy()[start_joint_index:end_joint_index]

    def get_pair(self):
        joint_rand_q = np.zeros(self.joint_dim)
        joint_rand_q[self.start_joint_index:self.end_joint_index] =  np.random.uniform(self.joint_limit_lb, self.joint_limit_ub)
        self.model.joint_q = wp.array(joint_rand_q,device=self.device, dtype=float)

        tape = wp.Tape()
        with tape:
            
            wp.sim.eval_fk(
                self.model,
                self.model.joint_q,
                self.model.joint_qd,
                None,
                self.state)

        x = self.state.body_q.numpy()[-1]
        
        qx = np.concatenate((joint_rand_q[self.start_joint_index:self.end_joint_index], x),dtype=np.float32)

        return qx




class KinematicsDataset(Dataset):
    def __init__(self, robot, len_batch= 4096):
        self.len_batch = len_batch
        self.robot = robot
    
    def __getitem__(self, index):
        qx = self.robot.get_pair()
        return qx

    def __len__(self):
        return self.len_batch
    


class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, dataloader):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.iters = 0
        
    def forward(self, x):
        zero = torch.zeros(x.shape[0], 1).to(x)
        x = x[:7]
        c = x[7:]
        return self.model(x,c,zero)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1
        x= batch
        
        zero = torch.zeros(x.shape[0], 1).to(x)

        c = x[:, 7:]
        x = x[:, :7]

        z, delta_logp = self.model(x, c, zero)

        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)

        return {'loss': loss} 
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-5)

    def train_dataloader(self):
        return self.dataloader


def get_robot(filepath):

    r = Robot(robot_path=filepath)
    qx = r.get_pair()

    dataset = KinematicsDataset(r)
    dataloader = DataLoader(dataset, batch_size=4096)
    
    return r, dataloader


def run():
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots', 'panda_arm.urdf')

    r, dataloader = get_robot(filepath)
    model = build_model_tabular_suhan(args, 7).to(device)
    learn = Learner(model, dataloader)

    trainer = pl.Trainer(max_epochs=1000000000,accelerator='gpu', devices=1)
    trainer.fit(learn)


if __name__ == '__main__':

    run()
    