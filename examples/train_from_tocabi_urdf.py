from dataclasses import dataclass

import os
# import pdb; # pdb.set_trace()

os.environ['WANDB_API_KEY']='7fa46452ab048b6302357208d967486b045b4808'

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot
from nodeik.training.datasets import KinematicsDataset
from nodeik.training.learner import Learner

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
    gpu = 0
    rademacher = False
    model_checkpoint = 'lightning_logs/epoch=195822-step=195823.ckpt'

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

wp.init()
import numpy as np
def get_robot(filepath):

    r = Robot(robot_path=filepath,end_joint_index=-1, ee_link_index=12)

    joint_q = np.zeros(r.joint_dim)
    print('self.joint_dim',r.joint_dim)

    fk = r.get_forward_kinematics_all(joint_q)

    renderer = wp.sim.render.SimRenderer(r.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_fk_grad.usd"))
    render_time = 0.0
    for _ in range(90):
        renderer.begin_frame(render_time)
        renderer.render(r.state)
        renderer.end_frame()
        render_time += 1.0/30.0
    renderer.save()

    # print('fk',fk)
    # print('fk:12',fk[12])
    # import pdb; pdb.set_trace()
    val_size = 256
    dataset = KinematicsDataset(r, len_batch=4096*500)
    val_dataset = KinematicsDataset(r, len_batch=val_size)
    dataloader = DataLoader(dataset, batch_size=4096)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size)
    
    return r, dataloader, val_dataloader

def run():
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots','tocabi', 'tocabi.urdf')
    r, dataloader, val_dataloader = get_robot(filepath)
    model = build_model(args, r.joint_dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    print('parameters', params)
    learn = Learner(model, robot=r, std=1.0,num_samples=4)
    learn.model_wrapper.device = 'cuda:0'
    # learn = Learner.load_from_checkpoint(args.model_checkpoint, model=model, robot=r, std=0.5)
    #epoch=158400-step=158401.ckpt

    wandb_logger = WandbLogger(project="node-ik", name='tocabi-corrected-1024-4-gpuserver', log_model='all')
    print(wandb_logger.version)
    print(wandb_logger)
    trainer = pl.Trainer(max_epochs=1000000000,accelerator='gpu', devices=1, gpus=[0], logger=wandb_logger, check_val_every_n_epoch=1, log_every_n_steps=10, default_root_dir='/home/ubuntu/sh_ws/nodeik/checkpoints')
    trainer.fit(learn,dataloader,val_dataloader)


if __name__ == '__main__':

    run()
    