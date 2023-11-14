import time
from typing import Callable, Optional
from ssvr.directions import DirectionMatrix
import torch
from ssvr.optimizer.gvr_zd import GVR_ZD

class SPIDER_ZD(GVR_ZD):
    
    
    def optimize(self, x0, T, p, gamma : float, h: Callable, verbose=False):
        x_k, x_prev = x0.clone(), None
        f_values, it_times = [], []
        for k in range(T):
            it_time = time.time()
            h_k = h(k)
            if k % p == 0:                
                P = self.P_full()
                f_full = self.target(x_k, None)
                v_k = self._build_g(x_k, fx = f_full, h=h_k, z=None, P=P)
            else:
                P = self.P()
                z = torch.randint(0, high=self.n, size=(1, ), generator=self.generator)
                g_k = self._build_g(x_k, fx = self.target(x_k, z), h = h_k, P = P, z =z)
                g_prev = self._build_g(x_prev, fx = self.target(x_prev, z), h = h_k, P = P, z =z)
                v_k = g_k - g_prev + v_k
            x_prev = x_k.clone()
            x_k = self._project(x_k - gamma * v_k)
            it_time = time.time() - it_time
            f_values.append(self.target(x_k))
            it_times.append(it_time)
            if verbose:
                print("[SPIDER-ZD] k = {}/{}\tf(x_k) = {}".format(k, T, f_values[-1]))
        return dict(x=x_k, f_values=f_values, it_times=it_times)       
            
