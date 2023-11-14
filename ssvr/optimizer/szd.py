import time
import torch

from math import sqrt
from typing import Callable, Optional
from ssvr.optimizer import GVR_ZD
from ssvr.directions import DirectionMatrix

class SZD(GVR_ZD):

    def __init__(self, 
                 target: Callable[[torch.Tensor, Optional[int]], float], 
                 n: int, 
                 P: DirectionMatrix, 
                 seed: int = 12131415, 
                 bounds: Optional[torch.Tensor] = None) -> None:
        super().__init__(target, n, P, None, seed, bounds)
    
    def optimize(self, x0, T, gamma: Callable, h: Callable, verbose = False):
        x_k = x0.clone()
        f_values, it_times = [], []
        for k in range(T):
            it_time = time.time()
            P_k = self.P()
            z = torch.randint(0, high=self.n, size=(1,), generator=self.generator)
            f_k = self.target(x_k, z = z)
            g_k = self._build_g(x_k, fx = f_k, h = h(k), P = P_k, z = z)
            x_k = self._project(x_k - gamma(k, x_k, f_k) * g_k)
            it_time = time.time() - it_time
            f_values.append(self.target(x_k).item())
            it_times.append(it_time)
            if verbose:
                print("[SZD] k = {}/{}\tf(x_k) = {}".format(k, T, f_values[-1]))
        return dict(x=x_k, f_values=f_values, it_times=it_times)