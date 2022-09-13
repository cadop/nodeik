import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import copy

from nodeik.utils import standard_normal_logprob

class ModelWrapper:
    def __init__(self, model, robot, device, std=1.0, dim_x = 7, dim_c = 7):
        self.model = model
        self.robot = robot
        self.device = device
        self.std = std
        self.dim_c = dim_c
        self.dim_x = dim_x

    def inverse_kinematics(self, pose, num_samples=1, z=None):
        
        device = self.device
        # device = 'cuda:1'
        pose = np.array(pose,dtype=np.float32)
        if len(pose.shape) == 1:
            c = np.repeat([pose], repeats=num_samples, axis=0)
        else:  # 2
            assert (len(pose.shape) == 2)
            c = np.repeat(pose, repeats=num_samples, axis=0)

        zero = torch.zeros(c.shape[0], 1).to(device)
        c = torch.from_numpy(c).to(device)

        if z is None:
            x = torch.normal(mean=0.0, std=self.std, size=(c.shape[0],self.dim_x)).to(device)
        elif isinstance(z, float):
            x = z * torch.ones(size=(c.shape[0], self.dim_x))
        elif isinstance(z, np.ndarray):
            if len(z.shape) == 2 and z.shape[0] != 1:
                x = torch.from_numpy(z).float().to(device)
                pass
            elif len(z.shape) == 1 or (len(z.shape) == 2 and z.shape[0] == 1):
                z_torch = torch.from_numpy(z).float().to(device)
                x = z_torch.repeat(c.shape[0], 1)
            else:
                assert(False)
        elif isinstance(z,torch.Tensor):
            if len(z.shape) == 2 and z.shape[0] != 1:
                z.float().to(device)
                pass
            elif len(z.shape) == 1 or (len(z.shape) == 2 and z.shape[0] == 1):
                x = z.repeat(c.shape[0], 1)
            else:
                assert(False)

        else:
            assert(False)
        
        ik_q, delta_logp = self.model(x,c,zero, rev=True)
        ik_q = ik_q.cpu().detach().numpy()
        return ik_q, delta_logp

    def forward_kinematics(self, q):
        if len(q.shape) == 2:
            x = []
            for joint in q:
                x.append(copy.deepcopy(self.robot.get_forward_kinematics(joint)))
            x = np.array(x, dtype=np.float32)
        else:
            x = self.robot.get_forward_kinematics(q)  
        return x

    def eval(self):
        self.model.eval()
        self.model.chain[0].odefunc.odefunc.calc_density = False