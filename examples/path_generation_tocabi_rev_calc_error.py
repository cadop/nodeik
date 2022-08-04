from dataclasses import dataclass
from fileinput import filename
from math import cos, inf, sin

import os
import copy
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot, RobotTocabi
from nodeik.training.datasets import KinematicsDataset
from nodeik.training.learner import Learner

from pyquaternion import Quaternion


os.environ['ADJOINT'] = '0'
os.environ['WANDB_API_KEY']='7fa46452ab048b6302357208d967486b045b4808'

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
    seed = 0
    num_samples = 500
    adjoint=False

np.random.seed(args.seed)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

wp.init()

def get_robot(filepath):

    r = RobotTocabi(robot_path=filepath, ee1_link_name='L_Wrist2_Link', ee2_link_name='R_Wrist2_Link')

    joint_q = np.zeros(r.active_joint_dim)
    print('self.joint_dim',r.joint_dim)
    

    fk = r.get_forward_kinematics_all(joint_q)
    print(fk)
    fk = r.get_forward_kinematics_all(joint_q)
    print('l r', )

    custom_q = np.array([0,0,0, 
                -0.38, -0.1, 0.99, 1.56, 1.50, 1.04, 1.30,0.0,
                0.62, 0.14, -0.56, 0.98,2.18, -0.72, 0.0, 0.0])
    fk_2 = r.get_forward_kinematics(custom_q)
    print('res: ', fk_2)

    # renderer = wp.sim.render.SimRenderer(r.model, os.path.join(os.path.dirname(__file__), "outputs/custom_q.usd"))
    # render_time = 0.0
    # for _ in range(90):
    #     renderer.begin_frame(render_time)
    #     renderer.render(r.state)
    #     renderer.end_frame()
    #     render_time += 1.0/30.0
    # renderer.save()

    # print('fk',fk)
    # print('fk:12',fk[12])
    # import pdb; pdb.set_trace()
    val_size = 10
    dataset = KinematicsDataset(r, len_batch=512*100)
    val_dataset = KinematicsDataset(r, len_batch=val_size)
    dataloader = DataLoader(dataset, batch_size=512)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size)
    
    return r, dataloader, val_dataloader


def run():
    import wandb
    run = wandb.init()
    artifact = run.use_artifact('psh117/node-ik/model-3d3f5n2x:v46', type='model')
    artifact_dir = artifact.download()
    print(artifact_dir)
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots','tocabi_description', 'dyros_tocabi.urdf')
    r, dataloader, val_dataloader = get_robot(filepath)
    print(r.active_joint_dim)
    model = build_model(args, r.active_joint_dim, condition_dims=14).to(device)
    # model = build_model(args, r.joint_dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    print('parameters', params)


    custom_q = np.array([0,0,0, 
                -0.38, -0.1, 0.99, 1.56, 1.50, 1.04, 1.30,0.0,
                0.62, 0.14, -0.56, 0.98,2.18, -0.72, 0.0, 0.0], dtype=np.float32)
    fk_2 = r.get_forward_kinematics(custom_q)
    fk_2 = np.array(fk_2, dtype=np.float32)
    print('res: ', fk_2)
    c_ref = torch.from_numpy(fk_2[None,:]).to(device)
    zero = torch.zeros(1, 1).to(device)
    model.eval()
    model.chain[0].odefunc.odefunc.calc_density = False
    q_ref = torch.from_numpy(custom_q[None,:]).to(device)

    z_ref, _ = model(q_ref, c_ref, zero, rev=False)
    x_dist = 0.7
    y_dist = 0.3
    z_dist = 0.5
    q_left = [0.5, 0.5, 0.5, -0.5]
    q_right = [0.5, 0.5, -0.5, 0.5]
    targets = []
    radius = 0.15
    for angle in np.linspace(0,2*np.pi, args.num_samples, endpoint=False):
        target_pose_left = np.array([x_dist+radius*cos(angle), y_dist, z_dist+radius*sin(angle)] + q_left,dtype=np.float32)
        target_pose_right = np.array([x_dist-radius*cos(angle), -y_dist, z_dist-radius*sin(angle)] + q_right,dtype=np.float32)
        x_target = np.concatenate((target_pose_left, target_pose_right),dtype=np.float32)
        targets.append(x_target)
    targets = np.array(targets,dtype=np.float32)
    # x_target = np.array([[x_dist, y_dist, z_dist,  0.707, 0.0, -0.707, 0.0,
    #                      x_dist, -y_dist, z_dist, 0.5, 0.5, -0.5, 0.5]], dtype=np.float32)
    # learn = Learner(model, robot=r, std=1.0, state_dim=r.active_joint_dim, condition_dim=14, num_samples=250)
    learn = Learner.load_from_checkpoint(artifact_dir + '/model.ckpt', model=model, robot=r, std=1.0, state_dim=r.active_joint_dim, condition_dim=14, num_samples=250)
    learn.model_wrapper.device = 'cuda:0'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # c = np.repeat(x_target,args.num_samples,axis=0)
    def inference_ik_q(targets, z_variable):
        c = targets
        c = torch.from_numpy(c).to(device)
        # z = torch.normal(mean=0.0, std=0.0, size=(c.shape[0],r.active_joint_dim)).to(device)
        z = z_variable.repeat(c.shape[0],1)
        # import pdb; pdb.set_trace()
        zero = torch.zeros(c.shape[0], 1).to(device)
        model.eval()
        model.chain[0].odefunc.odefunc.calc_density = False
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            t = time.time()
            ik_q, delta_logp = model.chain[0](z,c,zero, rev=True)
            t_end = time.time()
        print(t_end-t)
        ik_q = ik_q.cpu().detach().numpy()
        return ik_q


    def is_path_continuous(qs):
        thhold = 0.1
        for i in range(len(qs)):
            ifnorm = np.linalg.norm(qs[i] - qs[i-1],ord=inf) 
            if ifnorm > thhold:
                return False

        return True

    def createDirectory(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create the directory.")

    def render(sub_dir, file_name, qs):
        createDirectory(f'outputs/{sub_dir}')
        render_time = 0.0 
        renderer = wp.sim.render.SimRenderer(r.model, os.path.join(os.path.dirname(__file__), "outputs/{}/{}.usd".format(sub_dir,file_name)))
        for q in qs:
            r.get_forward_kinematics_all(q)
            renderer.begin_frame(render_time)
            renderer.render(r.state)
            renderer.end_frame()
            render_time += 1.0/5.0
        renderer.save()

    trials = 100
    from tqdm import tqdm
    std=0.1
    ik_q = inference_ik_q(targets, z_ref)


    errors = {'p_err_1':[], 'p_err_2':[], 'q_err_1':[], 'q_err_2':[]}
        # print(p_err_1, p_err_2)

    # render(f'0_new_tocabi_{x_dist}_{y_dist}_{z_dist}_{radius}_{std}_{args.seed}_rev', f'tocabi_for_ref', ik_q)

    for t in tqdm(range(trials)):
        z = torch.normal(mean=0.0, std=std, size=(1,r.active_joint_dim)).to(device)
        ik_q = inference_ik_q(targets, z)
        if is_path_continuous(ik_q):
            for q, target in zip(ik_q,targets):
                fk = r.get_forward_kinematics(q)
                p_err_1 = fk[:3] - target[:3]
                p_err_2 = fk[7:10] - target[7:10]
                q1 = Quaternion(array=fk[3:7])
                q2 = Quaternion(array=target[3:7])
                # print(q)
                q_err_1 = Quaternion.distance(q1, q2)
                q1 = Quaternion(array=fk[10:14])
                q2 = Quaternion(array=target[10:14])
                q_err_2 = Quaternion.distance(q1, q2)
                errors['p_err_1'].append(p_err_1)
                errors['p_err_2'].append(p_err_2)
                errors['q_err_1'].append(q_err_1)
                errors['q_err_2'].append(q_err_2)

    print(np.linalg.norm(errors['p_err_1'],axis=1).mean())
    print(np.linalg.norm(errors['p_err_2'],axis=1).mean())
    print(np.array(errors['q_err_1']).mean())
    print(np.array(errors['q_err_2']).mean())
    import pickle
    pickle.dump(errors,open('error_data_tocabi_rev2.pkl','wb'))
            # render(f'0_new_tocabi_{x_dist}_{y_dist}_{z_dist}_{radius}_{std}_rev', f'tocabi_rev_{t}', ik_q)
    exit()
if __name__ == '__main__':

    run()
    