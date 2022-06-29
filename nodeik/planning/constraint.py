# Not used this file yet, but the kinpy can be one way for importing the URDF file I guess
# So I brought it from my old codes.

from __future__ import print_function, division

import numpy as np

from math import atan2

import kinpy as kp
import quaternion
import numdifftools as nd

class DualArmConstraint:
    def __init__(self) -> None:
        self.name_lists_left = ['panda_left_joint{0}'.format(i) for i in range(1,8)]
        self.name_lists_top = ['panda_top_joint{0}'.format(i) for i in range(1,8)]
        self.joint_names = self.name_lists_left + self.name_lists_top
        self.chain = kp.build_chain_from_urdf(open('model.urdf','rb').read())

        self.tolerance = 1e-5
        self.max_iteration = 1000

    def calc_angle_error(self, q1, q2):
        q1 = quaternion.as_quat_array(q1)
        q2 = quaternion.as_quat_array(q2)
        # q2_conj[0] = q2[3]
        # q2_conj[1] = -q2[0]
        # q2_conj[2] = -q2[1]
        # q2_conj[3] = -q2[2]
        d = q1 * q2.conjugate()
        d = quaternion.as_float_array(d)
        # print ('dddd')
        # print(d)
        # print (atan2(np.linalg.norm(d[1:]), abs(d[0])))
        return 2 * atan2(np.linalg.norm(d[1:]), abs(d[0]))

    def function(self, q):
        joints = dict(zip(self.joint_names, q))
        transforms = self.chain.forward_kinematics(joints)
        lh = transforms['panda_left_hand']
        th = transforms['panda_top_hand']
        cur_chain = lh.inverse() * th
        # print(cur_chain)
        init_chain_pos = np.array([0, 0, 0.6])
        init_chain_quat = np.array([0, 1, 0, 0])

        pos_err = np.linalg.norm(cur_chain.pos - init_chain_pos)
        rot_err = self.calc_angle_error(cur_chain.rot, init_chain_quat) 
        # print(pos_err,rot_err)
        return np.array([pos_err, rot_err])

    def svd_solve(self, A, b):
        U, sigma, VT = np.linalg.svd(A)
        Sigma = np.zeros(A.shape)
        Sigma[:2,:2] = np.diag(sigma)
        # (U.dot(Sigma).dot(VT) - A).round(4)
        Sigma_pinv = np.zeros(A.shape).T
        Sigma_pinv[:2,:2] = np.diag(1/sigma[:2])
        x_svd = VT.T.dot(Sigma_pinv).dot(U.T).dot(b)
        return x_svd

    def project(self, q):
        jaco_fn = nd.Jacobian(self.function)
        # err = self.function(q)
        # print('j',j)
        # print('e',err)
        # # dq = np.linalg.solve(j,err)
        # print(dq)
        # err = self.function(q)
        # print(err)
        for _ in range(self.max_iteration):
            err = self.function(q)
            if np.linalg.norm(err) < self.tolerance:
                break
            j = jaco_fn(q)
            dq = self.svd_solve(j,err)
            q = q - dq  
            print(err)
            print(q)