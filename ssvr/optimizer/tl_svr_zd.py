import time
from typing import Callable, Optional
from ssvr.directions import DirectionMatrix
import torch
from ssvr.optimizer.gvr_zd import GVR_ZD

class TL_SVR_ZD(GVR_ZD):
    
    
    def optimize(self, x0, T, p, tau, gamma : float, h: Callable, option='random', verbose=False):
        x_k = x0.clone()
        w_k = x0.clone()
        f_values, it_times = [], []
        recompute = True
        for k in range(T):
            it_time = time.time()
            h_k = h(k)
            if recompute:
                g_full = self._build_g(w_k, fx = self.target(w_k, None), h=h_k, z=None, P=self.P_full())                
            z = torch.randint(0, high=self.n, size=(2,), generator=self.generator)
            P_k = self.P()
            g_k = self._build_g(x_k, fx=self.target(x_k, z[0]), h=h_k, P=P_k, z=z[0])
            g_w = self._build_g(w_k, fx=self.target(w_k, z[0]), h=h_k, P=P_k, z=z[0])
            g_szd = self._build_g(x_k, fx=self.target(x_k, z[1]), h=h_k, P=P_k, z=z[1])
            g = tau * g_szd + (1 - tau) * (g_k - g_w + g_full)
            x_k = self._project(x_k - gamma * g)
            if torch.rand(1, generator=self.generator).item() <= p:
                w_k = x_k
                recompute = True
            else:
                recompute = False
            it_time = time.time() - it_time
            f_values.append(self.target(x_k).item())
            it_times.append(it_time)
            if verbose:
                print("[TL-SVR-ZD] k = {}/{}\tf(x_k) = {}".format(k, T, f_values[-1]))
        return dict(x=x_k, f_values=f_values, it_times=it_times)       
            


    
    # def __init__(self, P_full : Optional[DirectionMatrix] = None, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    #     assert P_full is None or (P_full.d == self.P.d and P_full.l == self.P.l)
    #     self.P_full = self.P if P_full is None else P_full
    
    # def optimize(self, x0, T, m, gamma : float, h: Callable[[int], float]):
    #     for tau in range(T):
            
    #         for k in range(m):
                
    
    # def optimize(self, x0, T, gamma = 1.0, h = lambda t : 1e-5, p =0.5, verbose=False, out_file = None):
    #     x_tau = x0.clone()
    #     f_values, it_times = [self.fun(x_tau)], [0.0]
    #     if out_file is not None:
    #         out_f = open(out_file, 'w')
    #     if verbose:
    #         print("[L-SVRZ] f(x_0) = {}".format(f_values[0]))
    #     num_fevals = 0
    #     compute_full = True
    #     x_k = x_tau.copy()
    #     for tau in range(T):
    #         it_time = time.time()
    #         z = torch.randint(0, high=self.n, size=(1,)).item()
    #         if compute_full:
    #             P_tau, h_tau = self.P_full(), h(tau)
    #             g_full = self._build_g(x_tau, z = None, P = P_tau, h = h_tau)
    #             num_fevals += (self.P_full.l + 1) * self.n

    #         P_k = self.P()
    #         g_k = self._build_g(x_k, z, P_k, h_tau)
    #         g_w = self._build_g(x_tau, z, P_k, h_tau)
    #         x_k = x_k - gamma * (g_k - g_w + g_full)
    #         if torch.rand() <= p:
    #             compute_full = True
    #             x_tau = x_k
    #         else:
    #             compute_full = False
    #         it_time = time.time() - it_time
    #         if verbose:
    #             print("[ZO-L-SVR] {}/{}\tf(x_k) = {}".format(tau, T, self.fun(x_k)))
    #         f_values.append(self.fun(x_tau))
    #         it_times.append(it_time)
    #         if out_file is not None:
    #             out_f.write("{},{}\n".format(f_values[-1], it_time))
    #     if out_file is not None:
    #         out_f.close()
    #     print("[ZO-L-SVR] fevals = {}".format(num_fevals))
    #     return  self._build_result(x_tau, f_values[-1], f_values, it_times)