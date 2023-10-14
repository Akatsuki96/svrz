import time
import numpy as np

from ssvr.optimizer.szd import SZD

class SVRSZD(SZD):
    
    
    def _build_v(self, x_k, x_tau, z_k, P_k, h_tau, g_full):
        g_tau = self._build_g(x_tau, z_k, P_k, h_tau)            
        g_k = self._build_g(x_k, z_k, P_k, h_tau)
        return g_k - g_tau + g_full
    
    
    def optimize(self, x0, m, T, gamma, h, verbose=False):
        x_tau = x0.copy()
        x_k = x0.copy()
        x_traj = np.empty((m, x_k.shape[0]))
        x_trace, f_values, it_times = [x_tau.copy()], [self.fun(x_tau)], []
        for tau in range(T):
            it_time = time.time()
            P_tau = self.P()
            h_tau = h(tau)
            g_full = self._build_g(x_tau, z = None, P = P_tau, h = h_tau)
            x_k = x_tau.copy()
            for k in range(m):
                P_k = self.P()
                z_k = self.rnd_state.choice(self.n, 1)[0]
                v_k = self._build_v(x_k, x_tau, z_k, P_k, h_tau, g_full)
                x_traj[k] = x_k
                x_k = x_k - gamma * v_k
            x_tau = np.mean(x_traj, axis=0)
            it_time = time.time() - it_time
            if verbose:
                print("[SVRZD] {}/{}\tf(x_k) = {}".format(tau, T, self.fun(x_k)))
            x_trace.append(x_tau.copy())
            f_values.append(self.fun(x_tau))
            it_times.append(it_time)
        return  dict(x=x_trace[-1], fx=f_values[-1], f_values=f_values, x_trace=x_trace, it_times=it_times)