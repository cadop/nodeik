import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from nodeik.warp.urdf_loader import parse_urdf

class Robot():
    def __init__(self, render=True, num_envs=1, device='cpu', robot_path='panda_arm.urdf',start_joint_index=0, end_joint_index=7):
        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

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

        self.model = builder.finalize(device)
        self.model.ground = False
        self.device = device

        self.joint_dim = len(self.model.joint_q.numpy())
        self.start_joint_index = start_joint_index
        self.end_joint_index = end_joint_index
        self.joint_limit_lb = self.model.joint_limit_lower.numpy()[start_joint_index:end_joint_index]
        self.joint_limit_ub = self.model.joint_limit_upper.numpy()[start_joint_index:end_joint_index]
        
        self.state = self.model.state()
        joint_qdot = np.zeros(self.joint_dim)
        self.model.joint_qd = wp.array(joint_qdot,device=self.device, dtype=float)

    def get_forward_kinematics(self, q):
        self.model.joint_q = wp.array(q,device=self.device, dtype=float)
        
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        x = self.state.body_q.numpy()[-1] # TODO: change here (target link index)
        norm = np.linalg.norm(x)
        if norm > 10:
            print('something is wrong')
            print (norm)
            print('x',x)
            print('self.state.body_q.numpy()',self.state.body_q.numpy())
            assert(norm > 10)
        return x

    def get_pair(self):
        joint_rand_q = np.zeros(self.joint_dim)
        joint_rand_q[self.start_joint_index:self.end_joint_index] =  np.random.uniform(self.joint_limit_lb, self.joint_limit_ub)

        x = self.get_forward_kinematics(joint_rand_q)        
        qx = np.concatenate((joint_rand_q[self.start_joint_index:self.end_joint_index], x),dtype=np.float32)

        return qx
