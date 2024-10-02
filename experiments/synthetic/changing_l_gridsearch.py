import os
import sys
import torch
import numpy as np

from synthetic_utils import test_optimizer

sys.path.append("../")

from datasets import SyntheticDataset
from targets import LeastSquares
from utils import get_optimizer, get_cost_per_iter


dtype = torch.float64
device = 'cuda'
d = 50# 200
seed = 12131415


x_star = torch.tensor([0 for i in range(d)], device=device, dtype=dtype)


budget = 500000

h = lambda k : 1e-7
reps = 10

num_directions = [1, 10, 20, 30, 40, 50]
gammas = np.logspace(-4, 1, 30)
inner_iterations = [50, 100, 150]


out_path =  f"/data/mrando/svrz_results/changing_l"
os.makedirs(out_path + "/full_results", exist_ok=True)
os.makedirs(out_path + "/param_tuning", exist_ok=True)

mu, L = 1, 100

#for name in names:

name = "osvrz" #sys.argv[1]
dataset = SyntheticDataset(x_star=x_star, n = d, seed=seed, L = L, mu = mu)

target = LeastSquares(data=dataset, seed = seed)
x0 = torch.ones((1, d), dtype=dtype, device=device)
f0 = target(x0).item()


generator = torch.Generator(device=device).manual_seed(seed)
for l in num_directions:
    for m in inner_iterations:
        for gamma in gammas:
            exp_name = f"least_squares_{m}_{l}_{mu}_{L}"

            rnd_seed = torch.randint(0, 20000, size=(1,), device=device, generator=generator).cpu().item()
            opt = get_optimizer(name, d, l, dtype=dtype, device=device, seed=rnd_seed)
            cost_per_iter = get_cost_per_iter(name, d=d,l=l, m=m, n=d)
            T = budget // cost_per_iter if cost_per_iter is not None else budget
            opt_result = test_optimizer(target, optimizer=opt, x0 = x0.clone(), T = T, m=m, gamma=gamma, h=h, cost_per_iter=cost_per_iter, reps = reps)

            mu_vals, std_vals = opt_result['values']
            mu_time, std_time = opt_result['times']
            costs = opt_result['cost_per_iter']
            with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
                f.write(f"{name},{f0},{mu_vals[-1]},{std_vals[-1]},{mu_time[-1]},{std_time[-1]}\n")
            with open(f"{out_path}/full_results/{name}_{m}_{l}_{gamma}_{mu}_{L}.log", 'a') as f:
                for i in range(len(mu_vals)):
                    cost = costs[i] if isinstance(costs, list) else costs
                    f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost}\n")                
