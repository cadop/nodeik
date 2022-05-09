import argparse
import pickle
import numpy as np
import time

import torch
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl

from lib import build_model_tabular_suhan

SOLVERS = ["dopri5"]
parser = argparse.ArgumentParser('NodeIK')
parser.add_argument("--layer_type", type=str, default="concatsquash", choices=["concatsquash"])
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=False)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh")

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)

parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args([])

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.iters = 0

model = build_model_tabular_suhan(args, 7).to(device)
learn = Learner.load_from_checkpoint('model/panda_sample_model.ckpt',model=model)
model.eval()
model.chain[0].odefunc.odefunc.calc_density = False

input_pose = np.array([6.1946e-01, -1.6464e-02,  8.6722e-01,  4.7658e-01,  4.9979e-01,  7.2251e-01, -3.2554e-02])

max_len = 1
z = torch.normal(0, 1, size=(max_len, 7)).to(device)
c = torch.from_numpy(input_pose).float().to(device)
print('origin_c',c)
cc = torch.stack([c]*max_len).to(device)
zero = torch.zeros(z.shape[0], 1).to(z)

start_2 = time.time()
model.chain[0].odefunc.odefunc.calc_density = True
xx, delta_logp = model(z, cc, zero,reverse=True)
end_2 = time.time()
evals = model.chain[0].num_evals()
print('evals',evals)
print('after q', xx[:max_len,:])
print(delta_logp[:max_len,0])
print('time',(end_2 - start_2) * 1000,'ms')

input_pose = np.array([6.1946e-01, -1.6464e-02,  8.6722e-01,  4.7658e-01,  4.9979e-01,  7.2251e-01, -3.2554e-02])

max_len = 512
z = torch.normal(0, 1, size=(max_len, 7)).to(device)
c = torch.from_numpy(input_pose).float().to(device)
print('origin_c',c)
cc = torch.stack([c]*max_len).to(device)
zero = torch.zeros(z.shape[0], 1).to(z)

start_2 = time.time()
model.chain[0].odefunc.odefunc.calc_density = True
xx, delta_logp = model(z, cc, zero,reverse=True)
end_2 = time.time()
evals = model.chain[0].num_evals()
print('evals',evals)
# print('after q', xx[:max_len,:])
# print(delta_logp[:max_len,0])
print('time',(end_2 - start_2) * 1000,'ms')

input_pose = np.array([6.1946e-01, -1.6464e-02,  8.6722e-01,  4.7658e-01,  4.9979e-01,  7.2251e-01, -3.2554e-02])

max_len = 32768
z = torch.normal(0, 1, size=(max_len, 7)).to(device)
c = torch.from_numpy(input_pose).float().to(device)
print('origin_c',c)
cc = torch.stack([c]*max_len).to(device)
zero = torch.zeros(z.shape[0], 1).to(z)

start_2 = time.time()
model.chain[0].odefunc.odefunc.calc_density = True
xx, delta_logp = model(z, cc, zero,reverse=True)
end_2 = time.time()
evals = model.chain[0].num_evals()
print('evals',evals)
# print('after q', xx[:max_len,:])
# print(delta_logp[:max_len,0])
print('time',(end_2 - start_2) * 1000,'ms')