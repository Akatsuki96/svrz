import torch
import time

from ssvr.directions import DirectionMatrix

from typing import Optional, Callable

class GVR_ZD:
    
    def __init__(self, 
                 target : Callable[[torch.Tensor, Optional[int]], float], 
                 n :int, 
                 P : DirectionMatrix, 
                 P_full : Optional[DirectionMatrix] = None,
                 seed : int = 12131415,
                 bounds : Optional[torch.Tensor] = None,
                 ) -> None:
        self.target = target
        self.n = n
        self.P = P
        self.P_full = P if P_full is None else P_full
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(seed)
        self.bounds = bounds
        self.t = 1
        self.best = (None, torch.inf)

    def _build_g(self, x, fx, h, P, z = None):
        f_values : torch.Tensor = self.target(x + h * P.T, z=z)
        f_values.add_(fx, alpha=-1.0).div_(h)
        return (P @ f_values).T

    def _project(self, x : torch.Tensor):
        if self.bounds is None:
            return x
        return x.clip(min=self.bounds[:, 0], max=self.bounds[:, 1])
                                                                                 
    def optimize(self, x0, T, gamma : Callable, h : Callable):
        pass
   
   
# class TwoLoopsOpt(GVR_ZD):
    
    
#     def optimize(self, x0, T, m, gamma : float, h: Callable, option='random', verbose=False):
#         x_tau = x0.clone()
#         f_values, it_times = [], []

#         for tau in range(T):
#             it_time = time.time()
#             h_tau = h(tau)
#             P_tau = self.P_full()
#             #    def _build_g(self, x, fx, h, P, z = None):
#             f_full = self.target(x_tau, None)
#             g_full = self._build_g(x_tau, fx = f_full, h=h_tau, z=None, P=P_tau)
#             x_k = x_tau.clone()
#             x_trace = torch.empty((m, x0.shape[1]), dtype=self.P.dtype, device=self.P.device)
#             for k in range(m):
#                 z = torch.randint(0, high=self.n, size=(1,), generator=self.generator)
#                 P_k = self.P()
#                 f_k = self.target(x_k, z)
#                 f_tau = self.target(x_tau, z)
#                 g_k = self._build_g(x_k, fx=f_k, h=h_tau, P=P_k, z=z)
#                 g_tau = self._build_g(x_tau, fx=f_tau, h=h_tau, P=P_k, z=z)
#                 x_k = self._project(x_k - gamma * (g_k - g_tau + g_full))
#                 x_trace[k] = x_k.reshape(-1)
#             if option == 'random':
#                 x_tau = x_trace[torch.randint(0, high=m, size=(1, ), generator=self.generator), :]
#             elif option == 'average':
#                 x_tau = x_trace.mean(dim=0, keepdim=True)
#             else:
#                 raise NotImplementedError()
#             it_time = time.time() - it_time
#             f_values.append(self.target(x_tau))
#             it_times.append(it_time)
#             if verbose:
#                 print("[SVR-ZD] k = {}/{}\tf(x_k) = {}".format(tau, T, f_values[-1]))
#         return dict(x=x_tau, f_values=f_values, it_times=it_times)       
            
