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
d = 200
l = 25
seed = 12131415


x_star = torch.Tensor([0 for i in range(d)], device=device).to(dtype)
dataset = SyntheticDataset(x_star=x_star, n = d, seed=seed)
generator = torch.Generator()


x0 = torch.ones((1, d), dtype=dtype, device=device)

target = LeastSquares(data=dataset, seed = seed)

sszd_opt     = SSZD(P = QRDirections(d=d, l = l, seed = seed, device=device, dtype=dtype), nrm_const=d / l, seed=seed)
gauss_opt     = SSZD(P = GaussianDirections(d=d, l = l, seed = seed, device=device, dtype=dtype), nrm_const=1 /l, seed=seed)
sph_opt     = SSZD(P = SphericalDirections(d=d, l = l, seed = seed, device=device, dtype=dtype), nrm_const= d / l, seed=seed)

zosvrg_ave   = ZOSVRG(d = d, l = l, dtype = dtype, device = device, seed=seed, estimator='ave')
zosvrg_coord = ZOSVRG(d = d, l = l, dtype = dtype, device = device, seed=seed, estimator='coord')

szvr_g       = SZVR_G(d = d, l = l, dtype = dtype, device = device, seed = seed)
zosvrg_coord_rand = ZOSVRG_CoordRand(d = d, l = l, dtype = dtype, device = device, seed = seed)

spider_szo = SpiderSZO(d = d, l = l, dtype =dtype, device =device, seed = seed)
zo_spider_coord = ZOSpiderCoord(d = d, batch_size=1, dtype =dtype, device =device, seed = seed)

osvrz = OSVRZ(P = QRDirections(d = d, l = l, seed = seed, device = device, dtype = dtype), batch_size=1, seed=seed)
budget = 1000000

os.makedirs("./results/least_squares", exist_ok=True)

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
            
    with open(f"./results/least_squares/{name}.log", 'w') as f:
        for i in range(len(ris['values'][0])):
            cost = cost_per_iter[i] if isinstance(cost_per_iter, list) else cost_per_iter
            f.write(f"{ris['values'][0][i]},{ris['values'][1][i]},{ris['times'][0][i]},{ris['times'][1][i]},{cost}\n")                

    return None 


gamma = lambda k : 0.01 * (l/d) * (1/sqrt(k + 1))

h = lambda k : 1e-7#max(1e-5 / sqrt(k + 1), 1e-9)
reps = 10

m = 100



cost_per_iter = (4 * m + d * (l + 1))
T = budget // cost_per_iter
szvr_g_result = test_optimizer("szvr_g", szvr_g, x0, T, m,   0.00005,   h, cost_per_iter, reps = reps)

cost_per_iter = (2 * (l + 1) * m + d * (l + 1))
T = budget // cost_per_iter
zosvrg_ave_result = test_optimizer("zosvrg_ave",zosvrg_ave, x0, T, m,   0.0002,   h, cost_per_iter, reps = reps)


cost_per_iter = (4 * d * m + 2 * d * d)
T = budget // cost_per_iter
zosvrg_coo_result = test_optimizer("zosvrg_coord", zosvrg_coord, x0, T, m, 0.003, h, cost_per_iter, reps = reps)


cost_per_iter = (4 * l  * m + 2 * d * d )
T = budget // cost_per_iter
zosvrg_coord_rand_result = test_optimizer("zosvrg_coord_rand", zosvrg_coord_rand, x0, T, m, 0.0015, h, cost_per_iter, reps = reps)

T  = budget 
spider_szo_result = test_optimizer("spider_szo", spider_szo, x0, T, m, 0.001, h, None, reps = reps)

T  = budget 
zo_spider_coord_result = test_optimizer("zo_spider_coord", zo_spider_coord, x0, T, m, 0.0015, h, None, reps = reps)

cost_per_iter = (2 * (l + 1)  * m + d * (d + 1) )
T = budget // cost_per_iter
osvrz_result = test_optimizer("osvrz", osvrz, x0, T, m, 0.0015, h, cost_per_iter, reps = reps)

cost_per_iter = (l + 1)
T = budget // cost_per_iter
sszd_result = test_optimizer("sszd", sszd_opt, x0, T, None, gamma, h, cost_per_iter, reps = reps)

cost_per_iter = (l + 1)
T = budget // cost_per_iter
gauss_result = test_optimizer("gauss_opt", gauss_opt, x0, T, None, gamma, h, cost_per_iter, reps = reps)

cost_per_iter = (l + 1)
T = budget // cost_per_iter
sph_result = test_optimizer("sph_opt", sph_opt, x0, T, None, gamma, h, cost_per_iter, reps = reps)
