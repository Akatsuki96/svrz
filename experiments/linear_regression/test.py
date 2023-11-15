import torch
import numpy as np

from ssvr.optimizer import  SZD, SVRZD, LSVRZD, SAGAZD, SPIDER_ZD, TL_SVR_ZD, SARAH_ZD, SARAH_Plus_ZD
from ssvr.directions import QRDirections

from math import sqrt

d =  50
l = 20
n = 5000
device = 'cuda'
dtype= torch.float64
gen = torch.Generator(device=device)
gen.manual_seed(12131415)

def execute_experiment(optimizer, params, out_file, reps = 10):
    
    values = []
    it_times = []
    
    for _ in range(reps):
        res = optimizer.optimize(**params)
        values.append(res['f_values'])
        it_times.append(res['it_times'])
        
    values = np.array(values).reshape(reps, -1)
    it_times = np.array(it_times).reshape(reps, -1)
    it_times = np.cumsum(it_times, axis=1)
    mu_val, std_val = np.mean(values, axis=0), np.std(values, axis=0)
    mu_time, std_time = np.mean(it_times, axis=0), np.std(it_times, axis=0)
    print(mu_val.shape, mu_time.shape)
    with open(out_file, 'w') as f:
        for i in range(len(mu_val)):
            f.write("{},{},{},{}\n".format(mu_val[i], std_val[i], mu_time[i], std_time[i]))
            f.flush()
        
              
    
    

X =  torch.randn(n, d, generator=gen,dtype=dtype, device=device)
w_star = 20*torch.randn(d, generator=gen, dtype=dtype, device=device) + 10

y = X @ w_star
seed = 12131415
def target(w, z = None):
    if z is None:
        return torch.square(w @ X.T - y).mean(dim=1, keepdim=True)
   # print(torch.square(w @ X[z, :] - y[z]).shape)
    return torch.square(w @ X[z, :].T - y[z]).reshape(-1, 1)

w = torch.full((1, d), 12000.0, dtype=dtype, device=device)

P = QRDirections(d =d, l=l, seed=seed, device=device, dtype=dtype)

svr_zd_opt = SVRZD(target, n = n, P = P, seed = seed)
lsvr_zd_opt = LSVRZD(target, n = n, P = P, seed= seed)
szd_opt = SZD(target, n = n, P = P, seed= seed)
sagazd_opt = TL_SVR_ZD(target, n=n, P=P, seed=seed)
sarah_opt = SARAH_Plus_ZD(target, n=n, P=P, seed=seed)

gamma = lambda t, x, f : 0.1 *  (1/sqrt(t + 1))
h = lambda t : 1e-5#1e-3 * (l/d) * (1/(t + 1)**2)
m = 100
# svrg => stocahstic f evals = T * (n + 2*m ) * (l + 1) 
# szd => stochastic f evals = T * (l + 1)
T =  50
T2 = T  * (n + 2*m) # - n 
T3 = T * (n + 2*m) #budget // (l + 1)
#T4 = (T // m + 1) * n + (T - ((T // m + 1)))
reps = 10
verbose=True

budget = 38500

T = budget // (2 * l + 1) - budget // (10 * (l + 1) * n)

T = 10000

print("budget = ", (T // 100) * ((l + 1) * n)  + (2*l + 2) * (T - T//100))
budget = (T // m) * ((l + 1) * n)  + (2*l + 2) * (T - T//m)

T_svrg = budget // (n * (l + 1) + 2 * m * (l + 1))
T_szd = budget // (l + 1)
T_saga = T_szd - n * (l + 1)
# delta = (1 - ((2 *l + 2) // ((l + 1)*n - (2*l + 2))) )
# T_n = (budget // ((l + 1)*n - (2 * l +2))) * (1 // delta) #(budget * m) // ((l + 1) * n) - m *  (2*l + 2) * (T - T//100) /


##exit()
#execute_experiment(sagazd_opt, dict(x0=w.clone(), T=T_svrg, p = 0.5, tau=0.1, gamma=0.01, h=h, verbose=verbose), "tlsvrzd_results", reps = 1)
execute_experiment(sarah_opt, dict(x0=w.clone(), T=T_svrg, m=m, eta=1/8, gamma=0.01, h=h, verbose=verbose), "tlsvrzd_results", reps = 1)
# execute_experiment(lsvr_zd_opt, dict(x0=w.clone(), T=T_svrg, gamma=0.5, p=0.7, h=h, verbose=verbose), "lsvrzd_results", reps = reps)
# execute_experiment(svr_zd_opt, dict(x0=w.clone(), T=T_svrg, gamma=0.01, m=m, h=h, verbose=verbose), "svrzd_results", reps = reps)
#execute_experiment(szd_opt, dict(x0=w.clone(), T=T_szd, gamma=gamma, h=h, verbose=verbose), "szd_results", reps = reps)


# szd_svrg.optimize(w.clone(), T = T, m = m, option='average', gamma=0.05, h=h, verbose=True)
# szd_saga.optimize(w.clone(), T = T2, gamma=0.01, h=h, verbose=True)
# szd_opt.optimize(w.clone(), T = T3, gamma=gamma, h=h, verbose=True)
