import time
from typing import Callable, Optional
from ssvr.directions import DirectionMatrix
import torch
from ssvr.optimizer.gvr_zd import GVR_ZD

class SARAH_ZD(GVR_ZD):
    
    
    def optimize(self, x0, T, m, gamma : float, h: Callable, option='random', verbose=False):
        x_k, x_prev = x0.clone(), None
        f_values, it_times = [], []
        x_trace = torch.empty((m, x0.shape[1]), dtype=self.P.dtype, device=self.P.device)

        for tau in range(T):
            it_time = time.time()
            h_tau = h(tau)
            v_k = self._build_g(x_k, fx = self.target(x_k, None), h=h_tau, z=None, P=self.P_full())
            x_prev = x_k.clone()
            x_trace[0] = x_k.reshape(-1)
            x_k = self._project(x_k - gamma * v_k)
            for k in range(1, m):
                z = torch.randint(0, high=self.n, size=(1,), generator=self.generator)
                P_k = self.P()
                g_k = self._build_g(x_k, fx=self.target(x_k, z), h=h_tau, P=P_k, z=z)
                g_prev = self._build_g(x_prev, fx=self.target(x_prev, z), h=h_tau, P=P_k, z=z)
                v_k = g_k - g_prev + v_k
                x_prev = x_k.clone()
                x_k = self._project(x_k - gamma * v_k)
                x_trace[k] = x_k.reshape(-1)
            if option == 'random':
                x_k = x_trace[torch.randint(0, high=m, size=(1, ), generator=self.generator), :]
            elif option == 'average':
                x_k = x_trace.mean(dim=0, keepdim=True)
            else:
                raise NotImplementedError()
            it_time = time.time() - it_time
            f_values.append(self.target(x_k))
            it_times.append(it_time)
            if verbose:
                print("[SARAH-ZD] k = {}/{}\tf(x_k) = {}".format(tau, T, f_values[-1]))
        return dict(x=x_k, f_values=f_values, it_times=it_times)       
            

class SARAH_Plus_ZD(GVR_ZD):
    
    
    def optimize(self, x0, T, m, eta : float, gamma : float, h: Callable, option='random', verbose=False):
        x_k, x_prev = x0.clone(), None
        f_values, it_times = [], []
        x_trace = torch.empty((m, x0.shape[1]), dtype=self.P.dtype, device=self.P.device)

        for tau in range(T):
            it_time = time.time()
            h_tau = h(tau)
            v_k = self._build_g(x_k, fx = self.target(x_k, None), h=h_tau, z=None, P=self.P_full())
            x_prev = x_k.clone()
            x_trace[0] = x_k.reshape(-1)
            x_k = self._project(x_k - gamma * v_k)
            v_0_norm = torch.linalg.norm(v_k.reshape(-1), ord=2).square()
            k = 1
            while k < m and torch.linalg.norm(v_k.reshape(-1)).square() > eta * v_0_norm:
                z = torch.randint(0, high=self.n, size=(1,), generator=self.generator)
                P_k = self.P()
                g_k = self._build_g(x_k, fx=self.target(x_k, z), h=h_tau, P=P_k, z=z)
                g_prev = self._build_g(x_prev, fx=self.target(x_prev, z), h=h_tau, P=P_k, z=z)
                v_k = g_k - g_prev + v_k
                x_prev = x_k.clone()
                x_k = self._project(x_k - gamma * v_k)
                x_trace[k] = x_k.reshape(-1)
                k += 1
            if option == 'random':
                x_k = x_trace[torch.randint(0, high=k, size=(1, ), generator=self.generator), :]
            elif option == 'average':
                x_k = x_trace[:k, :].mean(dim=0, keepdim=True)
            else:
                raise NotImplementedError()
            it_time = time.time() - it_time
            f_values.append(self.target(x_k))
            it_times.append(it_time)
            if verbose:
                print("[SARAH+-ZD] k = {}/{}\tf(x_k) = {}".format(tau, T, f_values[-1]))
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
    #         g_tau = self._build_g(x_tau, z, P_k, h_tau)
    #         x_k = x_k - gamma * (g_k - g_tau + g_full)
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