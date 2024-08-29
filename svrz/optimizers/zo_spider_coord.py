
import tqdm
import torch
from time import time
from torch import Tensor
from typing import Dict, Callable

from svrz.utils import TargetFunction
from svrz.directions import DirectionGenerator, CoordinateDirections, SphericalDirections, GaussianDirections
from svrz.optimizers.abs_opt import AbsOptimizer

class ZOSpiderCoord(AbsOptimizer):

    def __init__(self, 
                 d : int,
                 batch_size : int,
                 dtype : torch.dtype = torch.float32,
                 device : str = 'cpu',
                 seed : int = 12131415):
        P = CoordinateDirections(d = d, l = d, dtype=dtype, seed=seed, device=device)
        P_full = CoordinateDirections(d = d, l = d, seed = seed, device = device, dtype = dtype)
        super().__init__(P = P, P_full = None, seed=seed)
        self.I = P_full()
        self.batch_size = batch_size
        
    def _approx_grad(self, f, x, z, fx, h):
        return f(x + h * self.I, z).add_(fx, alpha=-1).div_(h).mul(self.I).sum(dim=0, keepdims=True)



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
        c = 0
        l_values = [1]
        while k < T:
            iteration_time = time()
            h_k = h(c)
            if c % m == 0:
                g_full = self._approx_grad(f, x_k, None, f_k, h_k)
                v_k = g_full
                k += f.n * (self.P.d + 1)
                l_values.append(f.n * (self.P.d + 1))
            else:
                z_k = f.sample_z(self.batch_size)
                g_k = self._approx_grad(f, x_k, z_k, f(x_k, z_k), h_k)
                g_prev = self._approx_grad(f, x_prev, z_k, f(x_prev, z_k), h_k)
                v_k = g_k - g_prev + v_k
                k += 2 * self.batch_size *(self.P.d + 1)
                l_values.append(2 * self.batch_size *(self.P.d + 1))
            x_prev = x_k.clone()
            x_k = x_k - gamma * v_k
            f_k = f(x_k).flatten().item()
            iteration_time = time() - iteration_time
            f_values.append(f_k)
            it_times.append(iteration_time)
            print(f'k = {k}/{T}\tf_k = {f_values[-1]}\ttime = {iteration_time}')

            c+=1
        return dict(x = x_k, f_values = f_values, it_times = it_times, l_values = l_values)

