import torch
import sys
import os
import numpy as np

from bbclass_utils import test_optimizer, get_dataset

sys.path.append("../")
from targets import BBClassification
from utils import  get_optimizer, get_cost_per_iter


dtype = torch.float64
device = 'cuda'
seed = 123131415
generator = torch.Generator().manual_seed(seed)

dataset_name = sys.argv[1]

training_set, test_set = get_dataset(dataset_name, dtype=dtype, device=device, generator=generator)
target = BBClassification(dataset=training_set,seed=seed)
budget = 5000000
d = training_set.d
n = training_set.n

x0 = torch.ones((1, d), dtype=dtype, device=device)
f0 = target(x0).item()
h = lambda k : 1e-7

reps = 10

num_directions = [d // i for i in [10, 5, 4, 3, 2, 1]] 
gammas = np.logspace(-4, -1, 30)
names = ['osvrz', 'zosvrg_ave', 'zosvrg_coord', 'szvr_g', 'zosvrg_cr', 'zospider_szo', 'zospider_coord', 'sszd', 'gauss_fd', 'sph_fd']
out_path =  f"/data/mrando/svrz_results/black_box_class/{dataset_name}"
os.makedirs(out_path + "/full_results", exist_ok=True)
os.makedirs(out_path + "/param_tuning", exist_ok=True)
os.makedirs(out_path + "/test_errors", exist_ok=True)

for name in names:
    generator = torch.Generator(device=device).manual_seed(seed)
    for m in [100]:#[25, 50, 100]:
        for l in num_directions:
            for gamma in gammas:
                ris_path = f"{out_path}/{dataset_name}_{m}_{l}_{gamma}"
                exp_name = f"{dataset_name}_{m}_{l}_{gamma}"
                rnd_seed = torch.randint(0, 20000, size=(1,), device=device, generator=generator).cpu().item()
                opt = get_optimizer(name, d, l, dtype = dtype, device = device, seed = rnd_seed)
                cost_per_iter = get_cost_per_iter(name, d, l, m, n)
                T = budget // cost_per_iter if cost_per_iter is not None else budget
                opt_result = test_optimizer(target, opt, x0.clone(), T, m, gamma, h, test_set, cost_per_iter = cost_per_iter, reps = reps)
                mu_vals, std_vals = opt_result['values']
                mu_time, std_time = opt_result['times']
                mu_te, std_te = opt_result['test_error']
                costs = opt_result['cost_per_iter']
                with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
                    f.write(f"{name},{f0},{mu_vals[-1]},{std_vals[-1]},{mu_time[-1]},{std_time[-1]}\n")
                with open(f"{out_path}/test_errors/{exp_name}_te.log", 'a') as f:
                    f.write(f"{name},{mu_te},{std_te}\n")
                with open(f"{out_path}/full_results/{name}_{m}_{l}_{gamma}.log", 'a') as f:
                    for i in range(len(mu_vals)):
                        cost = costs[i] if isinstance(costs, list) else costs
                        f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost}\n")                
                    


