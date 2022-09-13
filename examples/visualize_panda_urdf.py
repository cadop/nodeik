from dataclasses import dataclass

import os
import copy
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader

from nodeik.utils import build_model, render, is_path_continuous

import warp as wp

from nodeik.robots.robot import Robot
from nodeik.training import KinematicsDataset, Learner, ModelWrapper

from pyquaternion import Quaternion

from math import cos, sin, pi
from tqdm import tqdm

@dataclass
class args:
    """
    std: standard deviation of prior distribution for z sampling
    num_trials: path generation trials
    resolution: path resolution (one circle)

    x_dist: x distance of TOCABI target
    y_dist: x distance of TOCABI target
    z_dist: x distance of TOCABI target
    q_left: Quaternion target for TOCABI left arm
    q_right: Quaternion target for TOCABI left arm
    radius: radius of target circle trajectory
    
    dt: delta time between frames
    """

    layer_type = 'concatsquash'
    dims = '1024-1024-1024-1024'
    num_blocks = 1 
    time_length = 0.5
    train_T = False
    divergence_fn = 'approximate'
    nonlinearity = 'tanh'
    solver = 'dopri5'
    atol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    rtol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    gpu = 0
    rademacher = False
    seed = 1
    model_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'model','panda_loss-20.ckpt')
    std = 1.0
    num_trials = 1000
    resolution = 1000

    x_dist = 0.5545
    y_dist = 0.0
    z_dist = 0.6245
    q = [1, 0, 0, 0]
    radius = 0.15

    dt = 1.0/5
    
np.random.seed(args.seed)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

wp.init()

def run():
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots','franka_panda', 'panda_arm.urdf')

    r = Robot(robot_path=filepath, ee_link_name='panda_hand')
    tasks = 1
    task_dim = 7 * tasks #  end-effector SE(3)
    model = build_model(args, r.active_joint_dim, condition_dims=task_dim).to(device)
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=model, robot=r, std=1.0, num_samples=250, state_dim=r.active_joint_dim, condition_dim=task_dim)
    learn.model_wrapper.device = device
    nodeik = learn.model_wrapper

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    ########################################
    # Target path generation
    radius = args.radius
    x_dist = args.x_dist
    y_dist = args.y_dist
    z_dist = args.z_dist
    q = args.q
    
    targets = []
    for angle in np.linspace(0,2*np.pi, args.resolution, endpoint=False):
        target_pos = np.array([x_dist, 
                               y_dist + radius * sin(angle),
                               z_dist + radius * cos(angle) - radius])
        target_pose = np.concatenate((target_pos, np.array(q)))
        targets.append(target_pose)
    targets = np.array(targets,dtype=np.float32)

    ########################################
    # Path IK solve and rendering
    std=args.std
    for t in tqdm(range(args.num_trials)):
        z = torch.normal(mean=0.0, std=std, size=(1,r.active_joint_dim)).to(device)
        ik_q, _ = nodeik.inverse_kinematics(targets, z=z)
        if is_path_continuous(ik_q):
            render(r, 
                   os.path.join(os.path.dirname(__file__), 'panda_renders'), 
                   f'panda_{x_dist}_{y_dist}_{z_dist}_{radius}_{args.seed}_{t}', 
                   ik_q, dt=1.0/5.0)

if __name__ == '__main__':

    run()
    