from dataclasses import dataclass

import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot
from nodeik.training import KinematicsDataset, Learner

from pyquaternion import Quaternion


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
    model_checkpoint = 'test.ckpt'

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
    model = build_model(args, 7).to(device)
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=model, dataloader=dataloader)
    model = learn.model
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pose_sets = []
    for i in range(args.num_references):
        qx = r.get_pair()
        x = qx[7:]
        pose_sets.append(x)

    c = np.array(pose_sets, dtype=np.float32)
    c = np.repeat(c,args.num_samples,axis=0)
    zero = torch.zeros(c.shape[0], 1).to(device)
    c = torch.from_numpy(c).to(device)

    # print('c',c)
    # x = torch.zeros(c.shape[0], 7).to(device)
    x = torch.normal(mean=0.0, std=0.7, size=(c.shape[0],7)).to(device)
    # print('x',x )
    # print(x.shape, c.shape, zero.shape)
    ik_q, delta_logp = model(x,c,zero, reverse=True)
    ik_q = ik_q.cpu().detach().numpy()

    # print('ik_q', ik_q)
    # print('delta_logp', delta_logp)

    ik_best_sets = []
    for i in range(args.num_references):
        max_delta_logp = -1e10
        best_q = np.zeros(7)

        for j in range(args.num_samples):
            cur_q = ik_q[i*args.num_samples + j]

            if (r.joint_limit_lb > cur_q).any():
                continue
            if (r.joint_limit_ub < cur_q).any():
                continue

            if delta_logp[i*args.num_samples + j] > max_delta_logp:
                max_delta_logp = delta_logp[i*args.num_samples + j]
                best_q = cur_q
        if max_delta_logp == -1e10:
            print('nothing fits')
            
        ik_best_sets.append((best_q, max_delta_logp))
    
    fk_sets = []
    for q, delta_logp in ik_best_sets:
        # print('q',q)
        # print('delta_logp',delta_logp)
        kf = r.get_forward_kinematics(q)
        # print('fk',kf)
        fk_sets.append(copy.deepcopy(kf))
        
    # print('pose_sets', pose_sets)
    # print('fk_sets', fk_sets)
    diff = np.array(pose_sets) - np.array(fk_sets)
    # print('diff', diff)

    p_err = []
    q_err = []
    for a, b in zip(pose_sets, fk_sets):
        pos_norm = np.linalg.norm(a[:3] - b[:3])
        q1 = Quaternion(array=a[3:])
        q2 = Quaternion(array=b[3:])
        quat_norm = Quaternion.distance(q1,q2)
        p_err.append(pos_norm)
        q_err.append(quat_norm)
    # for d in diff:
    #     pos_norm = np.linalg.norm(d[:3])
    #     quat_norm = np.linalg.norm(d[3:])
        # print(pos_norm,quat_norm)

    print('mean position    error:', np.array(p_err).mean())
    print('mean orientation error:', np.array(q_err).mean())
if __name__ == '__main__':

    run()
    