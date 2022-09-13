import os
import math
from numbers import Number
import logging
import six
import math
import nodeik.layers as layers
import torch
import numpy as np

from math import cos, inf, sin

import warp as wp


def isnan(tensor):
    return (tensor != tensor)

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def count_nfe(model):

    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):

    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def build_model(args, dims, condition_dims, regularization_fns=[]):

    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            dim_c=condition_dims,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            rademacher=args.rademacher,
        )
        cnf = layers.CCNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
            atol=args.atol,
            rtol=args.rtol,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    model = layers.SequentialFlow(chain)

    return model


def is_path_continuous(qs):
    thhold = 0.1
    for i in range(1,len(qs)):
        ifnorm = np.linalg.norm(qs[i] - qs[i-1],ord=inf) 
        if ifnorm > thhold:
            return False

    return True

def createDirectory(directory):
    print(directory)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('diretory is created:',directory)
    except OSError:
        print("Error: Failed to create the directory.")

def render(r, sub_dir, file_name, qs, dt):
    """
    r: Robot
    sub_dir: Directory
    file_name: Filename
    qs: Q set
    dt: delta time
    """
    createDirectory(f"{sub_dir}")
    render_time = 0.0 
    renderer = wp.sim.render.SimRenderer(r.model, "{}/{}.usd".format(sub_dir,file_name))
    for q in qs:
        r.get_forward_kinematics_all(q)
        renderer.begin_frame(render_time)
        renderer.render(r.state)
        renderer.end_frame()
        render_time += dt
    renderer.save()