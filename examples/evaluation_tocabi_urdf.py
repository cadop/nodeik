from dataclasses import dataclass

import os
import copy
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import RobotDual
from nodeik.training import KinematicsDataset, Learner, ModelWrapper

from pyquaternion import Quaternion

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
    atol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    rtol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    gpu = 0
    rademacher = False
    num_samples = 4
    num_references = 256
    seed = 1
    model_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'model','tocabi_dual_loss-6.ckpt')
    
np.random.seed(args.seed)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

wp.init()

def run():
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots','tocabi_description', 'dyros_tocabi.urdf')

    r = RobotDual(robot_path=filepath,
                  ee1_link_name='L_Wrist2_Link', 
                  ee2_link_name='R_Wrist2_Link', 
                  joint_map=[12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])

    tasks = 2
    task_dim = 7 * tasks #  end-effector SE(3)
    model = build_model(args, r.active_joint_dim, condition_dims=task_dim).to(device)
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=model, robot=r, std=1.0, num_samples=250, state_dim=r.active_joint_dim, condition_dim=task_dim)
    learn.model_wrapper.device = device
    nodeik = learn.model_wrapper
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pose_sets = []
    for _ in range(args.num_references):
        qx = r.get_pair()
        x = qx[r.active_joint_dim:]
        for _ in range(args.num_samples):
            pose_sets.append(x)

    c = np.array(pose_sets, dtype=np.float32)
    zero = torch.zeros(c.shape[0], 1).to(device)
    c = torch.from_numpy(c).to(device)
    x = torch.normal(mean=0.0, std=1.0, size=(c.shape[0],task_dim)).to(device)
    print(x.shape, c.shape, zero.shape)

    nodeik.eval()
    ik_q, _ = nodeik.inverse_kinematics(pose_sets)
    fk_sets = nodeik.forward_kinematics(ik_q)

    p_err = []
    q_err = []
    for a, b in zip(pose_sets, fk_sets):
        for i in range(tasks):
            a_loc = a[i*7:(i+1)*7]
            b_loc = b[i*7:(i+1)*7]
            pos_norm = np.linalg.norm(a_loc[:3] - b_loc[:3])
            q1 = Quaternion(array=a_loc[3:])
            q2 = Quaternion(array=b_loc[3:])
            quat_norm = Quaternion.distance(q1,q2)
            p_err.append(pos_norm)
            q_err.append(quat_norm)

    print('mean position    error:', np.array(p_err).mean())
    print('mean orientation error:', np.array(q_err).mean())
    
if __name__ == '__main__':

    run()
    