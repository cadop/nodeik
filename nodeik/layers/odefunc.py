import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from . import diffeq_layers

__all__ = ["ODEnet", "ODEfunc"]

def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self, hidden_dims, input_shape, conv, layer_type="concatsquash", nonlinearity="softplus", num_squeeze=0, dim_c = 7
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        for dim_out in hidden_dims + (input_shape[0],):
            layer = diffeq_layers.ConcatSquashLinear(hidden_shape[0], dim_out, dim_c=dim_c)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, tc, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(tc, dx)
            
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)

        return dx


class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))
        self.calc_density = True

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 3
        y = states[0]
        c = states[1]
        
        # Refresh the odefunc statistics.
        # _t1 = time.time()

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        # _t2 = time.time()

        if self.calc_density:
            t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
            with torch.set_grad_enabled(True):
                y.requires_grad_(True)
                c.requires_grad_(True)
                t.requires_grad_(True)
                for s_ in states[3:]:
                    s_.requires_grad_(True)
                    
                tc = torch.cat([t, c.view(y.shape[0], -1)], dim=1)
                
                dy = self.diffeq(tc, y, *states[3:])
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
                # if self.calc_density is True:
                #     # print('calc dnesioty')
                #     # Hack for 2D data to use brute force divergence computation.
                #     if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                #         divergence = divergence_bf(dy, y).view(batchsize, 1)
                #     else:
                #         divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
                # else: 
                #     divergence = torch.zeros_like(c[:,0]).requires_grad_(True)
                    # print('not calc dnesioty')
                # print('divergence.shape',divergence.shape)
            return tuple([dy, torch.zeros_like(c).requires_grad_(True), -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[3:]])

        else:
            with torch.no_grad():
                t = torch.ones(y.size(0), 1).to(y) * t.clone()#.detach().type_as(y)
                tc = torch.cat([t, c.view(y.shape[0], -1)], dim=1)
                dy = self.diffeq(tc, y, *states[3:])
                divergence = torch.zeros_like(c[:,0])

                return tuple([dy, torch.zeros_like(c), -divergence] + [torch.zeros_like(s_) for s_ in states[3:]])

        # print('times:',(_t2-_t1)*1000,(_t3-_t2)*1000,(_t4-_t3)*1000)
