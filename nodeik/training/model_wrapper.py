import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from nodeik.utils import standard_normal_logprob

class ModelWrapper:
    def __init__(self, model, robot, device, std=1.0):
        self.model = model
        self.robot = robot
        self.device = device
        self.std = std

    def inverse_kinematics(self, pose, num_samples=1):
        
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
        x = torch.normal(mean=0.0, std=self.std, size=(c.shape[0],7)).to(device)
        # import pdb; pdb.set_trace()
        ik_q, delta_logp = self.model(x,c,zero, reverse=True)
        ik_q = ik_q.cpu().detach().numpy()

        return ik_q, delta_logp

    def forward_kinematics(self, q):
        # import pdb; pdb.set_trace()
        if len(q.shape) == 2:
            x = []
            for joint in q:
                x.append(self.robot.get_forward_kinematics(joint))
            x = np.array(x, dtype=np.float32)
        else:
            x = self.robot.get_forward_kinematics(q)  
        return x
