import os
import sys
import torch
import numpy as np
from math import sqrt


from svrz import SSZD, ZOSVRG, SZVR_G, ZOSVRG_CoordRand, SpiderSZO, ZOSpiderCoord, OSVRZ
from svrz.directions import QRDirections, SphericalDirections, GaussianDirections


sys.path.append("../")

from datasets import SyntheticDataset
from targets import LeastSquares



dtype = torch.float64
device = 'cpu'
d = 50
seed = 12131415


x_star = torch.Tensor([0 for i in range(d)], device=device).to(dtype)
dataset = SyntheticDataset(x_star=x_star, n = d, seed=seed)
generator = torch.Generator()


x0 = torch.ones((1, d), dtype=dtype, device=device)

target = LeastSquares(data=dataset, seed = seed)

budget = 250000

os.makedirs("./results/changing_gamma", exist_ok=True)

def test_optimizer(name, optimizer, x0, T, m, gamma, h, l, cost_per_iter = None, reps = 10):
    values, times = [], []
    for _ in range(reps):
        if m is None:
            result = optimizer.optimize(target, x0 = x0, T = T, gamma = gamma, h = h)
        else:
            result = optimizer.optimize(target, x0 = x0, T = T, m = m, gamma = gamma, h = h)
        values.append(result['f_values'])
        times.append(result['it_times'])
    print("-"*44)
    values = np.array(values).reshape(reps, -1)
    times = np.array(times).reshape(reps, -1).cumsum(1)
    ris = {
        'values' : (values.mean(0), values.std(0)),
        'times' : (times.mean(0), times.std(0))
    }
    if cost_per_iter is None:
        cost_per_iter = result['l_values']
            
    with open(f"./results/changing_gamma/{gamma}_{m}_{l}.log", 'a') as f:
        f.write(f"{name},{target(x0).item()},{ris['values'][0][-1]},{ris['values'][1][-1]},{ris['times'][0][-1]},{ris['times'][1][-1]}\n")                

    return None 



h = lambda k : 1e-7#max(1e-5 / sqrt(k + 1), 1e-9)
reps = 10

generator = torch.Generator(device=device).manual_seed(seed)
num_directions = [1, 10, 25, 50, 75] #+ [i for i in range(10, d + 10, 10)]
gammas = np.logspace(-4, -1, 30)

for m in [25, 50, 75]:
    for l in num_directions:
        for gamma in gammas:
            rnd_seed = torch.randint(0, 20000, size=(1,), device=device, generator=generator).cpu().item()
#            osvrz = OSVRZ(P = QRDirections(d = d, l = l, seed = rnd_seed, device = device, dtype = dtype), batch_size=1, seed=rnd_seed)
            osvrz = ZOSVRG(d=d, l=l, batch_size=1, estimator='ave', dtype=dtype, device=device, seed=rnd_seed)
            cost_per_iter = (2 * (l + 1) * m + d * (l + 1))# (4 * m + d * (l + 1)) #(2 * (l + 1)  * m + d * (d + 1) )
            T = budget // cost_per_iter
            osvrz_result = test_optimizer(f"zo_svrg_ave", osvrz, x0, T, m, gamma, h, l, cost_per_iter, reps = reps)

