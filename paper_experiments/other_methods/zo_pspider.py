
import tqdm
import torch
from time import time
from torch import Tensor
from typing import Dict, Callable

from svrz.utils import TargetFunction
from svrz.optimizers.abs_opt import AbsOptimizer
from svrz.prox import ProxOperator

from .directions import SphericalDirections, CoordinateDirections

class ZOPSpider(AbsOptimizer):

    def __init__(self, 
                 d : int,
                 l : int,
                 prox : ProxOperator | None = None,
                 estimator : str = 'randsge',
                 dtype : torch.dtype = torch.float32,
                 device : str = 'cpu',
                 seed : int = 12131415):
        assert estimator in ['randsge', 'coosge']
        self.estimator = estimator
        if estimator == 'randsge':
            P = SphericalDirections(d = d, l = l, dtype=dtype, seed=seed, device=device)
        else:
            P = CoordinateDirections(d = d, l = d, dtype=dtype, seed=seed, device=device)
        P_full = CoordinateDirections(d = d, l = d, seed = seed, device = device, dtype = dtype)
        super().__init__(P = P, P_full = P_full, prox=prox, seed=seed)
        self.I = P_full()

        
    def _approx_grad(self, f, x, fx, h):
        return f(x + h * self.I).add_(f(x - h * self.I), alpha=-1).div_(2 * h).mul(self.I).sum(dim=0, keepdims=True)

    def _approx_sto_grad(self, f, x, z, fx, h, P):
        if self.estimator == 'randsge':
            return f(x + h * P, z, elem_wise=True).add_(fx, alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True).mul_(self.P.d / self.P.l)
        return f(x + h * P, z, elem_wise=True).add_(f(x - h * P, z, elem_wise=True), alpha=-1).div_(2 * h).mul(P).sum(dim=0, keepdims=True)


    def optimize(self, 
                 f : TargetFunction,  # objective function
                 x0: Tensor,  # initial guess
                 T: int, # number of outer iterations
                 m : int, # number of inner iterations
                 gamma : float, # stepsize 
                 h : Callable[[int], float], # smoothing parameter
                 callback : Callable[[Tensor, float, int], None] | None = None # callback
                 ) -> Dict:
        # f_values = [f(x0).flatten().item()]
        # it_times = [0.0]
        callback = callback if callback is not None else lambda x,t,iter:None
        x_prev = None
        x_k = x0.clone()
        f_k = f(x0).flatten().item() #f_values[-1]
        callback(x_k, 0.0, 0)
#        iterator = tqdm.tqdm(range(T))
        for tau in range(T):
            iteration_time = time()
            h_k = h(tau)
            v_prev =  self._approx_grad(f, x_k, f_k, h_k)
            x_prev = x_k.clone()
            for k in range(m):
                z_k = f.sample_z(self.P.l)
                P_k = self.P()
                g_k = self._approx_sto_grad(f, x_k, z_k, f(x_k, z_k, elem_wise=True), h_k, P_k)
                g_prev = self._approx_sto_grad(f, x_prev, z_k, f(x_prev, z_k, elem_wise=True), h_k, P_k)
                x_prev = x_k.clone()
                x_k = self.prox(x_k - gamma * (g_k - g_prev + v_prev), gamma=gamma)
                v_prev = g_k - g_prev + v_prev
            f_k = f(x_k).flatten().item()
            iteration_time = time() - iteration_time
            callback(x_k, iteration_time, tau + 1)

            # f_values.append(f_k)
            # it_times.append(iteration_time)
            # iterator.set_postfix({
            #     'k' : f"{tau}/{T}",
            #     'f_k' : f_values[-1],
            #     'time' : iteration_time
            # })

        return x_k #dict(x = x_k, f_values = f_values, it_times = it_times)

