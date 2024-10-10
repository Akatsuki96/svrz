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
d = 100
seed = 12131415


x_star = torch.tensor([0 for i in range(d)], device=device, dtype=dtype)


budget = 500000

h = lambda k : 1e-7
reps = 10

num_directions = [100, 80, 60, 50, 40, 20, 1] #[1] + [d // i for i in [10, 5, 4, 3, 2, 1]] + [100]
names = ['osvrz', 'zosvrg_ave', 'zosvrg_coord', 'szvr_g', 'zosvrg_cr', 'zospider_szo', 'zospider_coord', 'sszd', 'gauss_fd', 'sph_fd']

gammas = np.logspace(-4, -1, 50)
inner_iterations = [25, 50, 100]


out_path =  f"/data/mrando/svrz_results/synthetic_comp"
os.makedirs(out_path + "/full_results", exist_ok=True)
os.makedirs(out_path + "/param_tuning", exist_ok=True)

Ls = [10, 100, 1000]#, 1000]
L = int(sys.argv[2]) #100
mu = 1

#for name in names:

name = sys.argv[1]
#for L in Ls:
dataset = SyntheticDataset(x_star=x_star, n = d, seed=seed, L = L, mu = mu)

target = LeastSquares(data=dataset, seed = seed)
x0 = torch.ones((1, d), dtype=dtype, device=device)
f0 = target(x0).item()


num_dirs = num_directions if name not in ['zospider_coord', 'zosvrg_coord'] else [d]
for l in num_dirs:
    if name =='osvrz' and l > d:
        continue
    m_iters = inner_iterations if name not in ['sszd', 'gauss_fd', 'sph_fd'] else [None]
    for m in m_iters:
        for gamma in gammas:
            generator = torch.Generator(device=device).manual_seed(seed)
            exp_name = f"least_squares_{name}_{m}_{l}_{mu}_{L}"

            rnd_seed = torch.randint(0, 20000, size=(1,), device=device, generator=generator).cpu().item()
            opt = get_optimizer(name, d, l, dtype=dtype, device=device, seed=rnd_seed)
            cost_per_iter = get_cost_per_iter(name, d=d,l=l, m=m, n=d)
            T = budget // cost_per_iter if cost_per_iter is not None else budget
            opt_result = test_optimizer(target, optimizer=opt, x0 = x0.clone(), T = T, m=m, gamma=gamma, h=h, cost_per_iter=cost_per_iter, reps = reps)

            mu_vals, std_vals = opt_result['values']
            mu_time, std_time = opt_result['times']
            costs = opt_result['cost_per_iter']
            with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
                f.write(f"{name},{gamma},{f0},{mu_vals[-1]},{std_vals[-1]},{mu_time[-1]},{std_time[-1]}\n")
            with open(f"{out_path}/full_results/{name}_{m}_{l}_{gamma}_{mu}_{L}.log", 'a') as f:
                for i in range(len(mu_vals)):
                    cost = costs[i] if isinstance(costs, list) else costs
                    f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost}\n")                
