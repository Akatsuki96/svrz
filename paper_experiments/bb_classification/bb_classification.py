import os
import sys
import torch

from svrz.prox import SoftThreshold

sys.path.append("../")


from targets import BBClassification
from utils import get_optimizer, get_cost_per_iter, test_optimizer, get_dataset


dtype = torch.float64
device = 'cuda'
seed = 123131415


generator = torch.Generator().manual_seed(seed)

dataset_name = sys.argv[1]
datapath = sys.argv[2]

opt_name = sys.argv[3]
dataset = get_dataset(dataset_name, datapath , dtype=dtype, device=device, generator=generator)

target = BBClassification(dataset=dataset,seed=seed)
budget = 10000000 # => 1e7
d = dataset.d
n = dataset.n

if opt_name == 'rspgf':
    l = 1
elif opt_name == 'zo_pspider_coord' or opt_name == 'zo_psvrg_coord':
    l = d
else:
    l = d // int(sys.argv[4]) 
m = int(sys.argv[5])

out_path = sys.argv[6] + f"/{dataset_name}"

x0 = torch.zeros((1, d), dtype=dtype, device=device)
h = lambda k : 1e-5
b = 1
lam = 1e-5

print("n = {}\td = {}".format(n, d))
prox = SoftThreshold(lam=lam)

reps = 10

os.makedirs(out_path + "/full_results", exist_ok=True)
os.makedirs(out_path + "/param_tuning", exist_ok=True)



gammas =    [0.001, 0.01, 0.1, 1.0]
f0 = target.full_target(x0, elem_wise=False).item()

for (i, gamma) in enumerate(gammas):
    print(f"[{opt_name}] gamma = {gamma}")
    exp_name = f"{opt_name}_{l}_{m}_{b}"
    cost_per_iter = get_cost_per_iter(opt_name, d = d, l = l, m = m, n = d, b = b)
    num_iters = budget // cost_per_iter
    print(f"[{opt_name}] num iters = {num_iters}")
    opt = get_optimizer(name=opt_name, d = d, l = l, prox = prox, b = b, dtype = dtype, device = device, seed = seed)
    opt_result = test_optimizer(target=target, optimizer=opt, x0=x0, T = num_iters, m = m, gamma=gamma, h= h, reps=reps)

    mu_vals, std_vals = opt_result['values']
    mu_time, std_time = opt_result['times']

    if mu_vals[-1] != mu_vals[-1] or mu_vals[-1] + std_vals[-1] > f0:
        with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
            for j in range(len(gammas[i:])):
                f.write(f"{f0},{f0},{0.0},{mu_time[-1]},{std_time[-1]},{gammas[i + j]}\n")
        break
    else:
        with open(f"{out_path}/param_tuning/{exp_name}.log", 'a') as f:
            f.write(f"{f0},{mu_vals[-1]},{std_vals[-1]},{mu_time[-1]},{std_time[-1]},{gamma}\n")
        with open(f"{out_path}/full_results/{opt_name}_{l}_{m}_{b}_{gamma}.log", 'a') as f:
            for i in range(len(mu_vals)):
                f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost_per_iter}\n")     

