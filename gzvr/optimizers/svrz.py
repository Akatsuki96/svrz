from time import time
from gzvr.optimizers.abs_opt import Optimizer
from numpy import ndarray, empty, mean
from typing import Callable, Dict, Optional


class SVRZ(Optimizer):
    
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
        lst_evals = [1]
        x_tau = x0.copy()
        num_evals = 0
        for tau in range(1, T + 1):
            iteration_time = time()
            h_tau = h(tau)
            g, f = self._approximate_g(x_tau, P=self.P_full(), h = h_tau, z = None)
            x_k = x_tau.copy()
            x_trace = empty((m, self.P.d))
            num_evals += self.n * (self.P_full.l + 1)
            for k in range(m):
                z_k = self.rnd_state.choice(self.n, 1, replace=False)
                P_k = self.P()
                g_k, f_k = self._approximate_g(x_k, P=P_k, h = h_tau, z = z_k)
                g_tau, f_tau = self._approximate_g(x_tau, P = P_k, h = h_tau, z = z_k)
                x_k = x_k - gamma(k + 1, x_k, f_k) * (g_k - g_tau + g)
                x_trace[k] = x_k.copy()
                num_evals += 2 * (self.P.l + 1)
            if mode == 'average':
                x_tau = mean(x_trace, axis=0)
            elif mode == 'random':
                x_tau = x_trace[self.rnd_state.choice(m, size=1)].reshape(-1)
            elif mode =='last':
                x_tau = x_k
            iteration_time = time() - iteration_time
            f_values.append(self.f(x_tau))
            it_times.append(iteration_time)
            lst_evals.append(self.n * (self.P_full.l + 1) + 2 * m * (self.P.l + 1))
            if verbose:
                print("[SVRZ] k = {}/{}\tf(x_tau) = {}\ttime = {}s".format(k, T, f_values[-1], round(iteration_time, 4)))
        return dict(x = x_tau, f_values = f_values, lst_evals = lst_evals, it_times = it_times, num_evals = num_evals)
