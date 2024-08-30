import os
import sys
import torch
import numpy as np


from svrz import ZOSVRG, SZVR_G, ZOSVRG_CoordRand, SpiderSZO, ZOSpiderCoord, OSVRZ
from svrz.directions import QRDirections, SphericalDirections, GaussianDirections

from itertools import product

sys.path.append("../")

from datasets import SyntheticDataset
from targets import LeastSquares
from concurrent.futures import ProcessPoolExecutor as PPE


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
out_path = "/data/mrando/svrz_results/changing_params"

os.makedirs(out_path, exist_ok=True)

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
            
    with open(f"{out_path}/{name}.log", 'w') as f:
        for i in range(len(ris['values'][0])):
            cost = cost_per_iter[i] if isinstance(cost_per_iter, list) else cost_per_iter
            f.write(f"{ris['values'][0][i]},{ris['values'][1][i]},{ris['times'][0][i]},{ris['times'][1][i]},{cost}\n")                

    return None 


def get_optimizer(name, d, l, m, seed, device, dtype):
    if name == 'osvrz':
        return OSVRZ(P = QRDirections(d = d, l = l, seed = seed, device = device, dtype = dtype), batch_size=1, seed=seed), 2 * (l + 1)  * m + d * (d + 1) 
    elif name == 'szvr_g':
        return SZVR_G(d= d, l = l, dtype=dtype, device=device, seed=seed),  4 * m + d * (l + 1)
    elif name == 'zo_svrg_ave':
        return ZOSVRG(d = d, l=l, dtype=dtype, device=device, estimator='ave', batch_size=1, seed=seed), 2 * (l + 1) * m + d * (l + 1)
    elif name == 'zo_svrg_coord':
        return ZOSVRG(d = d, l=l, dtype=dtype, device=device, estimator='coord', batch_size=1, seed=seed), 4 * d * m + 2 * d * d
    elif name == 'zo_svrg_coord_rand':
        return ZOSVRG_CoordRand(d = d, l=l, dtype=dtype, device=device, seed=seed), 4 * l  * m + 2 * d * d 
    elif name == 'spider_szo':
        return SpiderSZO(d=d, l=l, dtype=dtype, device=device, seed=seed), None
    elif name == 'zo_spider_coord':
        return ZOSpiderCoord(d=d, batch_size=1, dtype=dtype, device=device, seed=seed), None


h = lambda k : 1e-7#max(1e-5 / sqrt(k + 1), 1e-9)
reps = 10

#m = 50
num_directions = [1, 5, 15, 25, 50] #i for i in range(5, d + 5, 5)]
gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
inner_iters = [5, 15, 25, 50]
opt_names = ['osvrz','szvr_g','zo_svrg_ave','zo_svrg_coord','zo_svrg_coord_rand','spider_szo', 'zo_spider_coord']


def execute_exp(param):
    name, l, gamma, m = param
    optimizer, cost_per_iter = get_optimizer(name, d, l, m, seed=seed, device=device, dtype=dtype)
    T = budget // cost_per_iter if cost_per_iter is not None else None
    print(f"[--] Executing {name}_{l}_{gamma}_{m}")
    opt_result = test_optimizer(f"{name}-{d}_{l}_{gamma}_{m}", optimizer, x0, T, m, gamma, h, cost_per_iter, reps = reps)
    return f"{name}-{d}_{l}_{gamma}_{m}"

max_workers = 4
num_completed = 0

grid = list(product(opt_names, num_directions, gammas, inner_iters))

with PPE(max_workers=max_workers) as executor:
    for result in executor.map(execute_exp, grid, chunksize= len(grid) // max_workers ):
        print(f"[--] Completed {result}")
        num_completed += 1
        
print(f"[++] Completed {num_completed} tasks!")
    

