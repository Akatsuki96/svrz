import os
import sys
import torch
import numpy as np


from svrz.prox import SoftThreshold

sys.path.append("../")

from datasets import SyntheticDataset
from targets import LeastSquares
from utils import get_optimizer, get_cost_per_iter, test_optimizer


dtype = torch.float64
device = 'cuda'
d = 50
seed = 12131415


x_star = torch.zeros((d, ), device=device, dtype=dtype)


budget = 100000

h = lambda k : 1e-7
reps = 10

gammas = np.logspace(-4, 0, 50)


out_path = str(sys.argv[4]) 
os.makedirs(out_path + "/full_results", exist_ok=True)
os.makedirs(out_path + "/param_tuning", exist_ok=True)


mu, L = 1, 10

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

    opt_result = test_optimizer(target, optimizer=opt, x0 = x0.clone(), T = T, m=m, gamma=gamma, h=h,  reps = reps)

    mu_vals, std_vals = opt_result['values']
    mu_time, std_time = opt_result['times']
    if mu_vals[-1] != mu_vals[-1] or mu_vals[-1] + std_vals[-1] > f0:
        with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
            for j in range(len(gammas[i:])):
                f.write(f"{f0},{f0},{0.0},{mu_time[-1]},{std_time[-1]}\n")
        break
    else:
        with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
            f.write(f"{f0},{mu_vals[-1]},{std_vals[-1]},{mu_time[-1]},{std_time[-1]}\n")
        with open(f"{out_path}/full_results/{name}_{m}_{l}_{b}_{gamma}_{mu}_{L}.log", 'a') as f:
            for i in range(len(mu_vals)):
                f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost_per_iter}\n")                
