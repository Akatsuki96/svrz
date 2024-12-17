import os
import sys
import torch
import numpy as np


from svrz.prox import SoftThreshold

sys.path.append("../")

from datasets import SyntheticDataset
from targets import LeastSquares
from utils import get_optimizer, get_cost_per_iter


dtype = torch.float64
device = 'cuda'
d = 50
seed = 12131415

class ResultCallback:
    
    def __init__(self, target, T, elem_wise = False, reps = 10):
        self.target = target
        self.elem_wise = elem_wise
        self.values = np.empty((reps, T + 1))
        self.it_times = np.empty((reps, T + 1))
        self.rep_counter = 0
        
    def __call__(self, x, it_time, tau):
        f_value = self.target.full_target(x, elem_wise=self.elem_wise)
        self.values[self.rep_counter, tau] = f_value
        self.it_times[self.rep_counter, tau] = it_time
        
    def go_next(self):
        self.rep_counter += 1

def test_optimizer(target, optimizer, x0, T, m, gamma, h, cost_per_iter, elem_wise = False, reps = 10):
    values, times = [], []
    cb = ResultCallback(target, T, elem_wise=elem_wise, reps=reps)
    for _ in range(reps):
        # if m is None:
        #     result = optimizer.optimize(target, x0 = x0, T = T, gamma = gamma, h = h, callback=cb)
        # else:
        result = optimizer.optimize(target, x0 = x0, T = T, m = m, gamma = gamma, h = h, callback=cb)
        cb.go_next()
        # values.append(result['f_values'])
        # times.append(result['it_times'])
    print("-"*44)
    values = cb.values# np.array(values).reshape(reps, -1)
    times = cb.it_times.cumsum(1)#np.array(times).reshape(reps, -1).cumsum(1)
    ris = {
        'values' : (values.mean(0), values.std(0)),
        'times' : (times.mean(0), times.std(0))
    }
#    if cost_per_iter is None:
#        cost_per_iter = result['l_values']
    ris['cost_per_iter'] = cost_per_iter           
    return ris


x_star = torch.zeros((d, ), device=device, dtype=dtype)


budget = 5000000

h = lambda k : 1e-7
reps = 10

gammas = np.logspace(-4, 0, 100)


out_path = str(sys.argv[4]) 
os.makedirs(out_path + "/full_results", exist_ok=True)
os.makedirs(out_path + "/param_tuning", exist_ok=True)


mu, L = 1, 100

name = "opsvrz"
dataset = SyntheticDataset(x_star=x_star, n = d, seed=seed, L = L, mu = mu)
lam = 1e-5
target = LeastSquares(data=dataset, lam=lam, seed = seed)
x0 = torch.ones((1, d), dtype=dtype, device=device)
f0 = target(x0).item()

l = int(sys.argv[1])
m = int(sys.argv[2])
b = int(sys.argv[3])
print(f"[--] Testing for l = {l}\tm = {m}\tb = {b}")
generator = torch.Generator(device=device).manual_seed(seed)
for (i, gamma) in enumerate(gammas):
    exp_name = f"{name}_{m}_{l}_{b}"

    rnd_seed = torch.randint(0, 20000, size=(1,), device=device, generator=generator).cpu().item()
    prox = SoftThreshold(lam = lam)
    opt = get_optimizer(name, d, l, prox=prox, b=b, dtype=dtype, device=device, seed=rnd_seed)
    cost_per_iter = get_cost_per_iter(name, d=d,l=l, m=m, n=d, b=b)
    T = budget // cost_per_iter if cost_per_iter is not None else budget
    opt_result = test_optimizer(target, optimizer=opt, x0 = x0.clone(), T = T, m=m, gamma=gamma, h=h, cost_per_iter=cost_per_iter, reps = reps)

    mu_vals, std_vals = opt_result['values']
    mu_time, std_time = opt_result['times']
    costs = opt_result['cost_per_iter']
    if mu_vals[-1] != mu_vals[-1] or mu_vals[-1] + std_vals[-1] > f0:
        with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
            for j in range(len(gammas[i:])):
                f.write(f"{name},{f0},{f0},{0.0},{mu_time[-1]},{std_time[-1]}\n")
        break
    else:
        with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
            f.write(f"{name},{f0},{mu_vals[-1]},{std_vals[-1]},{mu_time[-1]},{std_time[-1]}\n")
        with open(f"{out_path}/full_results/{name}_{m}_{l}_{b}_{gamma}_{mu}_{L}.log", 'a') as f:
            for i in range(len(mu_vals)):
                cost = costs[i] if isinstance(costs, list) else costs
                f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost}\n")                
