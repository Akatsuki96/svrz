import torch
from time import time
from torch import Tensor
from typing import Dict, Callable

from svrz.utils import TargetFunction
from svrz.prox import ProxOperator
from svrz.directions import DirectionGenerator
from svrz.optimizers.abs_opt import AbsOptimizer




class OPSVRZ(AbsOptimizer):

    def __init__(self, 
                 P : DirectionGenerator,  
                 batch_size : int = 1,
                 prox : ProxOperator | None = None,
                 seed : int = 12131415):
        super().__init__(P = P, P_full = None, prox = prox, seed=seed)
        self.nrm_const = P.d / P.l
        self.batch_size = batch_size
        self.I = torch.eye(P.d, dtype = P.dtype, device=P.device)
        
    def _approx_grad(self, f, x, z, fx, h, P):
        return f(x + h * P, z).add_(fx, alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True).mul_(P.shape[1] / P.shape[0])
        


    def optimize(self, 
                 f : TargetFunction,  # objective function
                 x0: Tensor,  # initial guess
                 T: int, # number of outer iterations
                 m : int, # number of inner iterations
                 gamma : float, # stepsize 
                 h : Callable[[int], float], # smoothing parameter
                 callback : Callable[[Tensor, float, int], None] | None = None # callback
                 ) -> Dict:
        callback = callback if callback is not None else lambda x,t,iter:None
#        f_values = [f(x0).flatten().item()]
#        it_times = [0.0]
        x_tau = x0.clone()
        f_tau = f(x0).flatten().item()
        callback(x_tau, 0.0, 0)
#        iterator = tqdm.tqdm(range(T))
        for tau in range(T):
            iteration_time = time()
            h_tau = h(tau)
            g_full = self._approx_grad(f, x_tau, None, f_tau, h_tau, self.I)
            x_k = x_tau.clone()
            for k in range(m):
                z_k = f.sample_z(self.batch_size)
                P_k = self.P()
                g_tau = self._approx_grad(f, x_tau, z_k, f(x_tau, z_k).flatten().item(), h_tau, P_k)
                g_k = self._approx_grad(f, x_k, z_k, f(x_k, z_k).flatten().item(), h_tau, P_k)
                x_k = self.prox(x_k - gamma * (g_k - g_tau + g_full), gamma)
            x_tau = x_k
            f_tau = f(x_tau).flatten().item()
            iteration_time = time() - iteration_time
            callback(x_tau, iteration_time, tau + 1)
#            f_values.append(f_tau)
#            it_times.append(iteration_time)
            # iterator.set_postfix({
            #     'k' : f"{tau}/{T}",
            #     'f_k' : f_values[-1],
            #     'time' : iteration_time
            # })
        return x_tau #dict(x = x_tau, f_values = f_values, it_times = it_times)

