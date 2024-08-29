
import tqdm
import torch
from time import time
from torch import Tensor
from typing import Dict, Callable

from svrz.utils import TargetFunction
from svrz.directions import DirectionGenerator, CoordinateDirections, SphericalDirections, GaussianDirections
from svrz.optimizers.abs_opt import AbsOptimizer

class SZVR_G(AbsOptimizer):

    def __init__(self, 
                 d : int,
                 l : int,
                 dtype : torch.dtype = torch.float32,
                 device : str = 'cpu',
                 seed : int = 12131415):
        P = GaussianDirections(d = d, l = l, dtype=dtype, seed=seed, device=device)
        super().__init__(P = P, P_full = None, seed=seed)
        self.nrm_const = 1 / P.l
        
    def _approx_full_grad(self, f, x, z, fx, h, P):
        return f(x + h * P, z).add_(fx, alpha=-1).div_(h).mul(P).sum(dim=0, keepdims=True).mul_(self.nrm_const)

    def _approx_sto_grad(self, f, x, z, fx, h, P):
        return f(x + h * P, z).add_(fx, alpha=-1).div_(h).mul(P)


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
        lst_evals = [1]
        num_evals = 0
        x_tau = x0.clone()
        f_tau = f_values[-1]
        iterator = tqdm.tqdm(range(T))
        for tau in iterator:
            iteration_time = time()
            h_tau = h(tau)
            P_tau = self.P()
            g_full = self._approx_full_grad(f, x_tau, None, f_tau, h_tau, P_tau)
            x_k = x_tau.clone()
            for k in range(m):
                z_k = f.sample_z()
                p_k = P_tau[torch.randint(0, self.P.l, size=(1,), generator=self.generator), :]
                
                g_tau = self._approx_sto_grad(f, x_tau, z_k, f(x_tau, z_k).flatten().item(), h_tau, p_k)
                g_k = self._approx_sto_grad(f, x_k, z_k, f(x_k, z_k).flatten().item(), h_tau, p_k)
                v_k = g_k - g_tau + g_full
                x_k = x_k - gamma * v_k
            x_tau = x_k
            f_tau = f(x_tau).flatten().item()
            iteration_time = time() - iteration_time
            f_values.append(f_tau)
            it_times.append(iteration_time)
            iterator.set_postfix({
                'k' : f"{tau}/{T}",
                'f_k' : f_values[-1],
                'time' : iteration_time
            })
        return dict(x = x_tau, f_values = f_values,  it_times = it_times)

