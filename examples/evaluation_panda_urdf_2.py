from dataclasses import dataclass

import os
import copy
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot
from nodeik.training import KinematicsDataset, Learner

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
    atol = 1e-5
    rtol = 1e-5
    gpu = 1
    rademacher = False
    num_samples = 16
    num_references = 512
    seed = 1
    model_checkpoint = '/home/ubuntu/sh_ws/nodeik/examples/node-ik/std=1-1024-4layers/checkpoints/epoch=39-step=20000.ckpt'
    # model_checkpoint = 'lightning_logs/epoch=158400-step=158401.ckpt'
    

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
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=model, robot=r)
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
    x = torch.normal(mean=0.0, std=0.3, size=(c.shape[0],7)).to(device)
    # print('x',x )
    print(x.shape, c.shape, zero.shape)
    model.eval()
    model.chain[0].odefunc.odefunc.calc_density = False

    for i in range(5):
        t_start = time.time()
        ik_q, delta_logp = model(x,c,zero, rev=True)
        import pdb; pdb.set_trace()
        t_end = time.time()
        ik_q = ik_q.cpu().detach().numpy()

        print('inference time!', t_end- t_start)

    # print('ik_q', ik_q)
    # print('delta_logp', delta_logp)

    ik_best_sets = []
    for i in range(args.num_references):
        max_delta_logp = -1e10
        best_q = np.zeros(5)

        for j in range(args.num_samples):
            cur_q = ik_q[i*args.num_samples + j]
            p = delta_logp[i*args.num_samples + j]

            # if (r.joint_limit_lb > cur_q).any():
            #     print('lb',cur_q)
            #     # print(p)
            #     continue
            # if (r.joint_limit_ub < cur_q).any():
            #     print('ub',cur_q)
            #     # print(p)
            #     continue
            
            if p > max_delta_logp:
                max_delta_logp = p
                best_q = cur_q
        if max_delta_logp == -1e10:
            print('nothing fits')
        # print(max_delta_logp)   
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
    # import pdb; pdb.set_trace()

    # print ('p_error')
    # print(p_err)
    
    # print ('q_error')
    # print(q_err)
    
if __name__ == '__main__':

    run()
    