import time
import numpy as np

from typing import Callable, Optional

from ssvr.directions import DirectionMatrix

class SZD:
    
    def __init__(self, 
                 fun : Callable[[np.ndarray, Optional[int]], float], 
                 d : int, 
                 l : int, 
                 n : int, 
                 P : DirectionMatrix, 
                 seed : int = 121314) -> None:
        self.fun = fun
        self.d = d
        self.l = l
        self.n = n
        self.P = P
        self.rnd_state = np.random.RandomState(seed = seed)
        
    def _build_g(self, x : np.ndarray, z : int, P : np.ndarray, h : float) -> np.ndarray:
        fx = self.fun(x, z)
        g = np.zeros(self.d)
        for i in range(P.shape[1]):
            g += ((self.fun(x + h *P[:, i], z) - fx)/h) * P[:, i]
        return g
        
    def optimize(self, 
                 x0 : np.ndarray, 
                 T : int, 
                 gamma = lambda t : 1/np.sqrt(t + 1),
                 h = lambda t : 1 / np.sqrt(t + 1),
                 verbose : bool = False
                 ):
        x_k = x0.copy()
        x_trace = [x0.copy()]
        f_values = [self.fun(x0)]
        it_times = []
        for k in range(T):
            it_time = time.time()
            P_k = self.P()
            gamma_k = gamma(k)
            h_k = h(k)
            z_k = self.rnd_state.choice(self.n, 1)[0]
            g_k = self._build_g(x_k, z_k, P_k, h_k)
            x_k = x_k - gamma_k * g_k
            it_time = time.time() - it_time
            if verbose:
                print("[SSZD] {}/{}\tf(x_k) = {}".format(k, T, self.fun(x_k)))
            x_trace.append(x_k.copy())
            f_values.append(self.fun(x_k))
            it_times.append(it_time)
        return dict(x=x_trace[-1], fx=f_values[-1], f_values=f_values, x_trace=x_trace, it_times=it_times)
            
        