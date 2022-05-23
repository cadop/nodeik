# SH Park: This is deprecated (May 23 2022)

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
import urdf_loader

wp.init()

TARGET = wp.constant(wp.vec3(.45, 0.0, 0.5))

@wp.kernel
def compute_loss(body_q: wp.array(dtype=wp.transform),
                 body_index: int,
                 loss: wp.array(dtype=float)):


    x = wp.transform_get_translation(body_q[body_index])

    delta = x - TARGET
    loss[0] = wp.dot(delta, delta)

@wp.kernel
def step_kernel(x: wp.array(dtype=float),
                grad: wp.array(dtype=float),
                alpha: float):

    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid]*alpha


class Robot:

    def __init__(self, render=True, num_envs=1, device='cpu'):


        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

        # wp.sim.parse_urdf(
        urdf_loader.parse_urdf(
            os.path.join(os.path.dirname(__file__), "panda_arm.urdf"), 
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

        # joint initial positions
        # builder.joint_q[-3:] = [0.0, 0.3, 0.0]
        
        # builder.joint_target[:3] = [0.0, 0.0, 0.0]

        # finalize model
        self.model = builder.finalize(device)
        # self.model.ground = True

        # self.model.joint_attach_ke = 1600.0
        # self.model.joint_attach_kd = 20.0

        # self.integrator = wp.sim.SemiImplicitIntegrator()

        # builder = wp.sim.ModelBuilder()

        # builder.add_articulation()

        # chain_length = 4
        # chain_width = 1.0

        # for i in range(chain_length):

        #     if i == 0:
        #         parent = -1
        #         parent_joint_xform = wp.transform([0.0, 0.0, 0.0], wp.quat_identity())           
        #     else:
        #         parent = builder.joint_count-1
        #         parent_joint_xform = wp.transform([chain_width, 0.0, 0.0], wp.quat_identity())

        #     joint_type = wp.sim.JOINT_REVOLUTE
        #     joint_axis=(0.0, 0.0, 1.0)
        #     joint_limit_lower=-np.deg2rad(60.0)
        #     joint_limit_upper=np.deg2rad(60.0)

        #     # create body
        #     b = builder.add_body(
        #             parent=parent,
        #             origin=wp.transform([i, 0.0, 0.0], wp.quat_identity()),
        #             joint_xform=parent_joint_xform,
        #             joint_axis=joint_axis,
        #             joint_type=joint_type,
        #             joint_limit_lower=joint_limit_lower,
        #             joint_limit_upper=joint_limit_upper,
        #             joint_target_ke=0.0,
        #             joint_target_kd=0.0,
        #             joint_limit_ke=30.0,
        #             joint_limit_kd=30.0,
        #             joint_armature=0.1)

        #     if i == chain_length-1:

        #         # create end effector
        #         s = builder.add_shape_sphere( 
        #                 pos=(0.0, 0.0, 0.0),
        #                 radius=0.1,
        #                 density=10.0,
        #                 body=b)

        #     else:
        #         # create shape
        #         s = builder.add_shape_box( 
        #                 pos=(chain_width*0.5, 0.0, 0.0),
        #                 hx=chain_width*0.5,
        #                 hy=0.1,
        #                 hz=0.1,
        #                 density=10.0,
        #                 body=b)




        # # finalize model
        # self.model = builder.finalize(device)
        self.model.ground = False

        self.device = device
        self.state = self.model.state()


        #-----------------------
        # set up Usd renderer
        self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_fk_grad.usd"))


    def run(self, render=True):

        render_time = 0.0
        train_iters = 1024
        train_rate = 0.01

        # optimization variables
        self.model.joint_q.requires_grad = True
        self.state.body_q.requires_grad = True

        self.loss = wp.zeros(1, dtype=float, device=self.device)
        print(self.model.joint_limit_lower)
        print(self.model.joint_limit_upper)
        print(self.model.joint_q)
        # print(self.model.links[0].name)
        # self.model.joint_q[3] += -math.pi/2
        self.model.joint_q = wp.array([0.0, 0.0, 0.0, -math.pi/2,0.0,math.pi/2,math.pi/4,0],device=self.device, dtype=float)

        for i in range(train_iters):

            tape = wp.Tape()
            with tape:
                
                wp.sim.eval_fk(
                    self.model,
                    self.model.joint_q,
                    self.model.joint_qd,
                    None,
                    self.state)

            k = self.state.body_q.numpy()

            print(k)
            fdas
            # print(k[-1])
            # print('---------------------')  
                # wp.launch(compute_loss, dim=1, inputs=[self.state.body_q, len(self.state.body_q)-1, self.loss], device=self.device)

            # tape.backward(loss=self.loss)

        #     print(' l',self.loss)
        #     print('dq',tape.gradients[self.model.joint_q])
        #     print(' q', self.model.joint_q)
            
        #     # gradient descent
        #     wp.launch(step_kernel, dim=len(self.model.joint_q), inputs=[self.model.joint_q, tape.gradients[self.model.joint_q], train_rate], device=self.device)

        #     # render
        #     self.renderer.begin_frame(render_time)
        #     self.renderer.render(self.state)
        #     self.renderer.render_sphere(name="target", pos=TARGET.val, rot=wp.quat_identity(), radius=0.1)
        #     self.renderer.end_frame()


        #     render_time += 1.0/60.0

        # self.renderer.save()
        

robot = Robot(render=False, device=wp.get_preferred_device(), num_envs=1)
robot.run()