
import tqdm
import torch
from time import time
from torch import Tensor
from typing import Dict, Callable
from math import sqrt

from svrz.prox import ProxOperator
from svrz.utils import TargetFunction
from svrz.optimizers.abs_opt import AbsOptimizer

from .directions import GaussianDirections

class RSPGF(AbsOptimizer):

    def __init__(self, 
                 d : int,
                 l : int = 1,
                 prox : ProxOperator | None = None,
                 seed : int = 12131415,
                 dtype :torch.dtype = torch.float32,
                 device : str = 'cpu'
                 ):
        P = GaussianDirections(d = d, l=l, seed=seed, device=device, dtype=dtype)
        super().__init__(P = P, P_full = None, prox=prox, seed=seed)
        self.nrm_const = 1/l
        
    def _approx_grad(self, f, x, z, fx, h):
        P_k = self.P()
        return f(x + h * P_k, z).add_(fx, alpha=-1).div_(h).mul(P_k).sum(dim=0, keepdims=True).mul_(self.nrm_const)
        

    def optimize(self, 
                 f : TargetFunction, 
                 x0: Tensor,  
                 T: int, 
                 m : int,
                 gamma : float, 
                 h : Callable[[int], float],
                 callback : Callable[[Tensor, float, int], None] | None = None # callback
                 ) -> Dict:
        # f_values = [f(x0).flatten().item()]
        # it_times = [0.0]
        # lst_evals = [1]
        # num_evals = 0
        x_k = x0.clone()
        # f_k = f(x0).flatten().item()
        # x_iters = [x_k]
        # iterator = tqdm.tqdm(range(T))
        callback(x_k, 0.0, 0)
        for k in range(T):
            iteration_time = time()
            z_k = f.sample_z()
            gamma_k, h_k = gamma / sqrt(k + 1), h(k)
            f_k = f(x_k, z_k).flatten().item()
            g_k = self._approx_grad(f, x_k, z_k, f_k, h_k)
            x_k = self.prox(x_k - gamma_k * g_k, gamma_k)
            iteration_time = time() - iteration_time
            callback(x_k, iteration_time, k + 1)
            # num_evals += self.P.l + 1
            # f_values.append(f(x_k).flatten().item())
            # it_times.append(iteration_time)
            # lst_evals.append(self.P.l + 1)
#             iterator.set_postfix({
# #                'x' : x_k,
#                 'k' : f"{k}/{T}",
#                 'f_k' : f"{f_values[-1]}",
#                 '|g_k|' : g_k.norm().item(),
#                 'time' : iteration_time
#             })
            # if k % 100 == 0:
            #     x_iters.append(x_k)
            
        return x_k # dict(x = x_k, f_values = f_values, lst_evals = lst_evals, it_times = it_times, num_evals=num_evals, x_iters=x_iters)
