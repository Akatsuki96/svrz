from typing import Dict, Callable, Optional
from numpy import ndarray
from gzvr.directions import DirectionGenerator
from gzvr.optimizers.abs_opt import Optimizer
from time import time

class SSZD(Optimizer):
    
    def __init__(self, 
                 f: Callable[[ndarray, int], float], 
                 n: int, 
                 P: DirectionGenerator, 
                 seed: int = 121314) -> None:
        super().__init__(f, n, P, None, seed)
    
    def optimize(self, 
                 x0: ndarray, 
                 T: int,
                 gamma : Callable[[int, Optional[ndarray], Optional[float]], float], 
                 h : Callable[[int], float],
                 verbose: bool = False) -> Dict:
        f_values = [self.f(x0)]
        it_times = [0.0]
        lst_evals = [1]
        num_evals = 0
        x_k = x0.copy()
        for k in range(1, T + 1):
            iteration_time = time()
            z_k = self.rnd_state.choice(self.n, 1, replace=False)
            h_k = h(k)
            g_k, f_k = self._approximate_g(x_k, self.P(), h_k, z = z_k)
            gamma_k = gamma(k, x_k, f_k)
            x_k = x_k - gamma_k * g_k
            iteration_time = time() - iteration_time
            num_evals += self.P.l + 1
            f_values.append(self.f(x_k))
            it_times.append(iteration_time)
            lst_evals.append(self.P.l + 1)
            if verbose:
                print("[SSZD] k = {}/{}\tf(x_k) = {}\ttime = {}s".format(k, T, f_values[-1], round(iteration_time, 4)))
        return dict(x = x_k, f_values = f_values, lst_evals = lst_evals, it_times = it_times, num_evals=num_evals)
