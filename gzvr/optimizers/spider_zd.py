from time import time
from gzvr.optimizers.abs_opt import Optimizer
from numpy import ndarray, empty, mean
from typing import Callable, Dict, Optional


class Spider_ZD(Optimizer):
    
    def optimize(self, 
                 x0: ndarray, 
                 T : int,
                 m : int, 
                 gamma: Callable[[int, Optional[ndarray], Optional[float]], float], 
                 h: Callable[[int], float], 
                 mode : str = 'average',
                 verbose: bool = False) -> Dict:
        assert  mode in ['average', 'random', 'last'], "Mode {} is not implemented!".format(mode)
        f_values = [self.f(x0)]
        it_times = [0.0]
        x_prev = None
        x_k = x0.copy()
        num_evals = 0
        lst_evals = [1]
        for k in range(T):
            iteration_time = time()
            h_k = h(k)
            if k % m ==0:
                g, f = self._approximate_g(x_k, P=self.P_full(), h = h_k, z = None)
                v_k = g
                num_evals += self.n * (self.P_full.l + 1)
                lst_evals.append(self.n * (self.P_full.l + 1))
            else:
                P_k = self.P()
                z_k = self.rnd_state.randint(0, high=self.n, size=1)
                g_k, f_k = self._approximate_g(x_k, P=P_k, h = h_k, z = z_k)
                g_prev, _ = self._approximate_g(x_prev, P = P_k, h = h_k, z = z_k)
                v_k = g_k - g_prev + v_k
                num_evals += 2 * (self.P.l + 1)
                lst_evals.append(2 * (self.P.l + 1))
            x_prev = x_k.copy()
            x_k = x_k - gamma * v_k
            iteration_time = time() - iteration_time
            f_values.append(self.f(x_k))
            it_times.append(iteration_time)
            if verbose:
                print("[SVRZ] k = {}/{}\tf(x_tau) = {}\ttime = {}s".format(k, T, f_values[-1], round(iteration_time, 4)))
        return dict(x = x_k, f_values = f_values, lst_evals=lst_evals, it_times = it_times, num_evals = num_evals)
