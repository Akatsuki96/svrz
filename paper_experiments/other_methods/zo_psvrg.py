import torch
from time import time
from torch import Tensor
from typing import Dict, Callable

from svrz.optimizers import AbsOptimizer
from svrz.utils import TargetFunction
from svrz.prox import ProxOperator

from .directions import CoordinateDirections, SphericalDirections, GaussianDirections


class ZOPSVRG(AbsOptimizer):
    
    def __init__(self, 
                 d : int,
                 l : int,
                 dir_type : str = 'spherical',
                 prox : ProxOperator | None = None,
                 dtype : torch.dtype = torch.float32,
                 device : str = 'cpu',
                 seed : int = 12131415):
        assert dir_type in ['spherical', 'gaussian', 'coordinate']
        if dir_type == 'spherical':
            P = SphericalDirections(d = d, l = l, dtype=dtype, device=device, seed = seed)
        elif dir_type == 'gaussian':
            P = GaussianDirections(d = d, l = l, dtype=dtype, device=device, seed = seed)
        else:
            P = CoordinateDirections(d = d, l = d, dtype=dtype, device=device, seed = seed)            
        self.dir_type = dir_type
        P_full = CoordinateDirections(d = P.d, l = P.d, seed = seed, device = P.device, dtype=P.dtype)
        super().__init__(P = P, P_full = P_full, prox=prox, seed=seed)
        
    def _approx_full_grad(self, f, x, h):
        P = self.P_full()
        return f(x + h * P).add_(f(x - h * P), alpha=-1).div_(2 *h).mul(P).sum(dim=0, keepdims=True)
        
    def _approx_sto_grad(self, f, x, z, fx, h, P):
        if self.dir_type == 'spherical': # or self.dir_type == 'gaussian':
            return f(x + h * P, z, elem_wise=True).add_(fx, alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True).mul_(self.P.d / self.P.l)
        elif self.dir_type == 'gaussian':
            return f(x + h * P, z, elem_wise=True).add_(fx, alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True).mul_(1 / self.P.l)
        return f(x + h * P, z).add_(f(x - h * P, z), alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True) #.mul_(self.P.d / self.P.l)
        

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
        x_tau = x0.clone()
        f_tau = f(x0).flatten().item()
        callback(x_tau, 0.0, 0)
#        iterator = tqdm.tqdm(range(T))
        for tau in range(T):
            iteration_time = time()
            h_tau = h(tau)
            g_full = self._approx_full_grad(f, x_tau, h_tau)
            x_k = x_tau.clone()
            for k in range(m):
                z_k = f.sample_z(self.P.l)
                P_k = self.P()
                f_tau, f_k = f(x_tau, z_k, elem_wise=True), f(x_k, z_k, elem_wise=True)
                g_tau = self._approx_sto_grad(f, x_tau, z_k, f_tau, h_tau, P_k)
                g_k = self._approx_sto_grad(f, x_k, z_k, f_k, h_tau, P_k)
                x_k = self.prox(x_k - gamma * (g_k - g_tau + g_full), gamma)
            x_tau = x_k
            f_tau = f(x_tau).flatten().item()
            iteration_time = time() - iteration_time
            callback(x_tau, iteration_time, tau + 1)
            # f_values.append(f_tau)
            # it_times.append(iteration_time)
            # iterator.set_postfix({
            #     'k' : f"{tau}/{T}",
            #     'f_k' : f_values[-1],
            #     'time' : iteration_time
            # })
        return x_tau #dict(x = x_tau, f_values = f_values, it_times = it_times)

