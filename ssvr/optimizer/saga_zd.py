import time
from typing import Callable, Optional
from ssvr.directions import DirectionMatrix
import torch
from ssvr.optimizer.gvr_zd import GVR_ZD

class SAGAZD(GVR_ZD):
    
    
    def optimize(self, x0, T, gamma : float, h: Callable, verbose=False):
        f_values, it_times = [], []
        P_full = self.P_full()
        grad_full = torch.empty((self.n, self.P.d), dtype=self.P.dtype) #
        phi = [x0.clone() for _ in range(self.n)]
        for i in range(self.n):
            grad_full[i] = self._build_g(x0, fx=self.target(x0, i), h = h(0), P=P_full, z = i).reshape(-1)
            
        x_k = x0.clone()
        for tau in range(T):
            it_time = time.time()
            h_tau = h(tau)
            z = torch.randint(0, high=self.n, size=(1,), generator=self.generator)
            P_k = self.P()
            g_k = self._build_g(x_k, fx=self.target(x_k, z), h=h_tau, P=P_k, z=z)
            g_tau = self._build_g(phi[z], fx=self.target(phi[z], z), h=h_tau, P=P_k, z=z)
            v_k = g_k - g_tau + grad_full.mean(dim=0).reshape(1, -1)
            x_k = self._project(x_k - gamma * v_k)
            phi[z] = x_k.clone()
            grad_full[z] = g_k.clone().reshape(-1)# - g_tau).reshape(-1)
            it_time = time.time() - it_time
            f_values.append(self.target(x_k))
            it_times.append(it_time)
            if verbose:
                print("[SAGA-ZD] k = {}/{}\tf(x_k) = {}".format(tau, T, f_values[-1]))
        return dict(x=x_k, f_values=f_values, it_times=it_times)       
            