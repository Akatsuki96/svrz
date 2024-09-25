
import tqdm
import torch
from time import time
from torch import Tensor
from typing import Dict, Callable

from svrz.utils import TargetFunction
from svrz.directions import DirectionGenerator, CoordinateDirections, SphericalDirections, GaussianDirections
from svrz.optimizers.abs_opt import AbsOptimizer

class SpiderSZO(AbsOptimizer):

    def __init__(self, 
                 d : int,
                 l : int,
                 dtype : torch.dtype = torch.float32,
                 device : str = 'cpu',
                 seed : int = 12131415):
        P = GaussianDirections(d = d, l = l, dtype=dtype, seed=seed, device=device)
        P_full = CoordinateDirections(d = d, l = d, seed = seed, device = device, dtype = dtype)
        super().__init__(P = P, P_full = None, seed=seed)
        self.I = P_full()
        
    def _approx_grad(self, f, x, fx, h):
        return f(x + h * self.I).add_(fx, alpha=-1).div_(h).mul(self.I).sum(dim=0, keepdims=True)

    def _approx_sto_grad(self, f, x, z, fx, h, P):
        return f(x + h * P, z, elem_wise=True).add_(fx, alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True).mul_(1/ self.P.l)


    def optimize(self, 
                 f : TargetFunction,  # objective function
                 x0: Tensor,  # initial guess
                 T: int, # number of outer iterations
                 m : int, # number of inner iterations
                 gamma : float, # stepsize 
                 h : Callable[[int], float] # smoothing parameter
                 ) -> Dict:
        f_values = [f(x0).flatten().item()]
        it_times = [0.0]
        x_prev = None
        x_k = x0.clone()
        f_k = f_values[-1]
        k = 0
        l_values = [1]
        c = 0 
        while k < T:
            iteration_time = time()
            h_k = h(c)
            if c % m == 0:
                g_full = self._approx_grad(f, x_k, f_k, h_k)
                v_k = g_full
                k += f.n * (self.P.d + 1)
                l_values.append(f.n * (self.P.d + 1))
            else:
                z_k = f.sample_z(self.P.l)
                P_k = self.P()
                g_k = self._approx_sto_grad(f, x_k, z_k, f(x_k, z_k, elem_wise=True), h_k, P_k)
                g_prev = self._approx_sto_grad(f, x_prev, z_k, f(x_prev, z_k, elem_wise=True), h_k, P_k)
                v_k = g_k - g_prev + v_k
                k += 4 * self.P.l
                l_values.append(4 * self.P.l)
            x_prev = x_k.clone()
            x_k = x_k - gamma * v_k
            f_k = f(x_k).flatten().item()
            iteration_time = time() - iteration_time
            f_values.append(f_k)
            it_times.append(iteration_time)
            c += 1
            print(f'k = {k}/{T}\tf_k = {f_values[-1]}\ttime = {iteration_time}')

        return dict(x = x_k, f_values = f_values, it_times = it_times, l_values=l_values)

