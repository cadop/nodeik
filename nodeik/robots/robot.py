import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from nodeik.warp.urdf_loader import parse_urdf

class Robot():
    def __init__(self, render=True, num_envs=1, device='cpu', robot_path='panda_arm.urdf',start_joint_index=0, end_joint_index=-1, ee_link_index=-1):
        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

        self.link_index = parse_urdf(
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

        if end_joint_index == -1: end_joint_index = self.joint_dim

        self.start_joint_index = start_joint_index
        self.end_joint_index = end_joint_index
        self.joint_limit_lb = self.model.joint_limit_lower.numpy()[start_joint_index:end_joint_index]
        self.joint_limit_ub = self.model.joint_limit_upper.numpy()[start_joint_index:end_joint_index]
        self.ee_link_index = ee_link_index
        
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

        x = self.state.body_q.numpy()[self.ee_link_index] # TODO: change here (target link index)
        norm = np.linalg.norm(x)
        if norm > 10:
            print('something is wrong')
            print (norm)
            print('x',x)
            print('self.state.body_q.numpy()',self.state.body_q.numpy())
            assert(norm > 10)
        return x

    def get_forward_kinematics_all(self, q):
        self.model.joint_q = wp.array(q,device=self.device, dtype=float)
        
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        x = self.state.body_q.numpy()
        
        return x

    def get_pair(self):
        joint_rand_q = np.zeros(self.joint_dim)
        joint_rand_q[self.start_joint_index:self.end_joint_index] =  np.random.uniform(self.joint_limit_lb, self.joint_limit_ub)

        x = self.get_forward_kinematics(joint_rand_q)        
        qx = np.concatenate((joint_rand_q[self.start_joint_index:self.end_joint_index], x),dtype=np.float32)

        return qx

class RobotDual():
    def __init__(self, render=True, num_envs=1, device='cpu', robot_path='dyros_tocabi.urdf',start_joint_index=0, ee1_link_name='L_Wrist2_Link', ee2_link_name='R_Wrist2_Link'):
        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

        self.link_index = parse_urdf(
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

        if end_joint_index == -1: end_joint_index = self.joint_dim

        self.start_joint_index = start_joint_index
        self.end_joint_index = end_joint_index
        self.joint_limit_lb = self.model.joint_limit_lower.numpy()[start_joint_index:end_joint_index]
        self.joint_limit_ub = self.model.joint_limit_upper.numpy()[start_joint_index:end_joint_index]
        self.ee_link_index_1 = self.link_index[ee1_link_name]
        self.ee_link_index_2 = self.link_index[ee2_link_name]
        
        self.state = self.model.state()
        joint_qdot = np.zeros(self.joint_dim)
        self.model.joint_qd = wp.array(joint_qdot,device=self.device, dtype=float)

    def get_forward_kinematics(self, q):
        self.model.joint_q = wp.array(q, device=self.device, dtype=float)
        
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        x = self.state.body_q.numpy()[self.ee_link_index] # TODO: change here (target link index)
        norm = np.linalg.norm(x)
        if norm > 10:
            print('something is wrong')
            print (norm)
            print('x',x)
            print('self.state.body_q.numpy()',self.state.body_q.numpy())
            assert(norm > 10)
        return x

    def get_forward_kinematics_all(self, q):
        self.model.joint_q = wp.array(q,device=self.device, dtype=float)
        
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        x = self.state.body_q.numpy()
        
        return x

    def get_pair(self):
        joint_rand_q = np.zeros(self.joint_dim)
        joint_rand_q[self.start_joint_index:self.end_joint_index] =  np.random.uniform(self.joint_limit_lb, self.joint_limit_ub)

        x = self.get_forward_kinematics(joint_rand_q)        
        qx = np.concatenate((joint_rand_q[self.start_joint_index:self.end_joint_index], x),dtype=np.float32)

        return qx



class RobotTocabi():
    def __init__(self, render=True, num_envs=1, device='cpu', robot_path='dyros_tocabi.urdf', ee1_link_name='L_Wrist1_Link', ee2_link_name='R_Wrist2_Link'):

        """
        0 R_HipYaw_Joint
        1 R_HipRoll_Joint
        2 R_HipPitch_Joint
        3 R_Knee_Joint
        4 R_AnklePitch_Joint
        5 R_AnkleRoll_Joint
        6 L_HipYaw_Joint
        7 L_HipRoll_Joint
        8 L_HipPitch_Joint
        9 L_Knee_Joint
        10 L_AnklePitch_Joint
        11 L_AnkleRoll_Joint
        12 Waist1_Joint
        13 Waist2_Joint
        14 Upperbody_Joint
        15 Neck_Joint
        16 Head_Joint
        17 L_Shoulder1_Joint
        18 L_Shoulder2_Joint
        19 L_Shoulder3_Joint
        20 L_Armlink_Joint
        21 L_Elbow_Joint
        22 L_Forearm_Joint
        23 L_Wrist1_Joint
        24 L_Wrist2_Joint
        25 R_Shoulder1_Joint
        26 R_Shoulder2_Joint
        27 R_Shoulder3_Joint
        28 R_Armlink_Joint
        29 R_Elbow_Joint
        30 R_Forearm_Joint
        31 R_Wrist1_Joint
        32 R_Wrist2_Joint
        """

        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs
        self.joint_map = [12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

        self.link_index = parse_urdf(
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
        self.active_joint_dim = len(self.joint_map)

        self.joint_limit_lb = self.model.joint_limit_lower.numpy()[self.joint_map]
        self.joint_limit_ub = self.model.joint_limit_upper.numpy()[self.joint_map]
        self.ee_link_index_1 = self.link_index[ee1_link_name]
        self.ee_link_index_2 = self.link_index[ee2_link_name]
        
        self.state = self.model.state()
        joint_qdot = np.zeros(self.joint_dim)
        self.model.joint_qd = wp.array(joint_qdot,device=self.device, dtype=float)

    def update_joint_q(self, q):
        joint_q = np.zeros(self.joint_dim)
        joint_q[self.joint_map] = q
        self.model.joint_q = wp.array(joint_q,device=self.device, dtype=float)

    def get_forward_kinematics_all(self, q, calc_collision=False):
        self.update_joint_q(q)
        
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        body_q = self.state.body_q.numpy() # TODO: change here (target link index)

        if calc_collision:
            self.model.collide(self.state)

        return body_q

    def get_forward_kinematics(self, q, calc_collision=False):
        body_q = self.get_forward_kinematics_all(q, calc_collision=calc_collision)
        return np.concatenate((body_q[self.ee_link_index_1], body_q[self.ee_link_index_2]))

    def get_pair(self):
        joint_rand_q = np.zeros(self.active_joint_dim)
        joint_rand_q = np.random.uniform(self.joint_limit_lb, self.joint_limit_ub)

        x = self.get_forward_kinematics(joint_rand_q)        
        qx = np.concatenate((joint_rand_q, x),dtype=np.float32)

        return qx

    def get_contact_num(self):
        return self.model.contact_count


class RobotTocabiSingle():
    def __init__(self, render=True, num_envs=1, device='cpu', robot_path='dyros_tocabi.urdf', ee_link_name='L_Wrist2_Link'):
        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs
        self.joint_map = [12,13,14,17,18,19,20,21,22,23,24]

        self.link_index = parse_urdf(
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
        self.active_joint_dim = len(self.joint_map)

        self.joint_limit_lb = self.model.joint_limit_lower.numpy()[self.joint_map]
        self.joint_limit_ub = self.model.joint_limit_upper.numpy()[self.joint_map]
        self.ee_link_index = self.link_index[ee_link_name]
        
        self.state = self.model.state()
        joint_qdot = np.zeros(self.joint_dim)
        self.model.joint_qd = wp.array(joint_qdot,device=self.device, dtype=float)

    def get_forward_kinematics_all(self, q):
        joint_q = np.zeros(self.joint_dim)
        joint_q[self.joint_map] = q
        self.model.joint_q = wp.array(joint_q,device=self.device, dtype=float)
        
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        body_q = self.state.body_q.numpy() # TODO: change here (target link index)

        return body_q

    def get_forward_kinematics(self, q):
        body_q = self.get_forward_kinematics_all(q)
        return body_q[self.ee_link_index]

    def get_pair(self):
        joint_rand_q = np.zeros(self.active_joint_dim)
        joint_rand_q = np.random.uniform(self.joint_limit_lb, self.joint_limit_ub)

        x = self.get_forward_kinematics(joint_rand_q)        
        qx = np.concatenate((joint_rand_q, x),dtype=np.float32)

        return qx
