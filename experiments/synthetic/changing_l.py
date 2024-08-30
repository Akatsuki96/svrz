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

budget = 1000000

os.makedirs("./results/changing_l", exist_ok=True)

def test_optimizer(name, optimizer, x0, T, m, gamma, h, cost_per_iter = None, reps = 10):
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
            
    with open(f"./results/changing_l/{name}.log", 'w') as f:
        for i in range(len(ris['values'][0])):
            cost = cost_per_iter[i] if isinstance(cost_per_iter, list) else cost_per_iter
            f.write(f"{ris['values'][0][i]},{ris['values'][1][i]},{ris['times'][0][i]},{ris['times'][1][i]},{cost}\n")                

    return None 



h = lambda k : 1e-7#max(1e-5 / sqrt(k + 1), 1e-9)
reps = 10

m = 50
num_directions = [1] + [i for i in range(5, d + 5, 5)]
for l in num_directions:
    osvrz = OSVRZ(P = QRDirections(d = d, l = l, seed = seed, device = device, dtype = dtype), batch_size=1, seed=seed)
    cost_per_iter = (2 * (l + 1)  * m + d * (d + 1) )
    T = budget // cost_per_iter
    osvrz_result = test_optimizer(f"osvrz_{l}", osvrz, x0, T, m, 0.001 * (l/d), h, cost_per_iter, reps = reps)

