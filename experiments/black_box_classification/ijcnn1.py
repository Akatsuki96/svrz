import torch
import sys
import os
import numpy as np

from torch.nn import BCELoss
from torch.utils.data import random_split

from bbclass_utils import test_optimizer

sys.path.append("../")
from datasets import RealDataset, load_libsvm_data
from targets import BBClassification
from utils import train_test_split, standardize

from math import sqrt

from svrz.optimizers import SSZD, OSVRZ, SpiderSZO, ZOSVRG_CoordRand, ZOSVRG, SZVR_G, ZOSpiderCoord
from svrz.directions import QRDirections, GaussianDirections, SphericalDirections


dtype = torch.float64
device = 'cuda'
training_fraction=0.8
seed = 123131415
generator = torch.Generator().manual_seed(seed)


X_tr, y_tr = load_libsvm_data(datapath="/data/mrando/ijcnn1/ijcnn1.tr", dtype=dtype, device=device)
X_te, y_te = load_libsvm_data(datapath="/data/mrando/ijcnn1/ijcnn1.t", dtype=dtype, device=device)


print(X_tr.shape, X_te.shape)
#exit()
training_set = RealDataset(*standardize(X_tr, y_tr))
test_set     = RealDataset(*standardize(X_te, y_te))


target = BBClassification(dataset=training_set,seed=seed)

d = training_set.d
l = d // 2


x0 = torch.ones((1, d), dtype=dtype, device=device)




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
budget = 5000000




reps = 2#10

h = lambda k : 1e-7

stepsize_const = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]) * (l / d)
inner_iters = [50, 100, 200]

gamma_c = 0.01
m = 100

# for gamma_c in stepsize_const:
#     for m in inner_iters:
        
out_path = f"/data/mrando/svrz_results/bb_class/ijcnn1/ijcnn1_{gamma_c}_{m}"
os.makedirs(out_path, exist_ok=True)

gamma = lambda k : gamma_c  * (1/sqrt(k + 1))

cost_per_iter = (2 * (l + 1)  * m + d * (d + 1) )
T = budget // cost_per_iter
osvrz_result = test_optimizer(target, "osvrz", osvrz, x0, T, m, gamma_c, h, test_set, cost_per_iter, reps=reps, out_path=out_path)

cost_per_iter = (4 * m + d * (l + 1))
T = budget // cost_per_iter
szvr_g_result = test_optimizer(target, "szvr_g", szvr_g, x0, T, m, gamma_c, h, test_set, cost_per_iter, reps=reps, out_path=out_path)

cost_per_iter = (2 * (l + 1) * m + d * (l + 1))
T = budget // cost_per_iter
zosvrg_ave_result = test_optimizer(target, "zosvrg_ave", zosvrg_ave, x0, T, m, gamma_c, h, test_set, cost_per_iter, reps=reps, out_path=out_path)

cost_per_iter = (4 * d * m + 2 * d * d)
T = budget // cost_per_iter
zosvrg_coo_result = test_optimizer(target, "zosvrg_coord", zosvrg_coord, x0, T, m, gamma_c, h, test_set, cost_per_iter, reps=reps, out_path=out_path)

cost_per_iter = (4 * l * m + 2 * d * d)
T = budget // cost_per_iter
zosvrg_coord_rand_result = test_optimizer(target, "zosvrg_coord_rand", zosvrg_coord_rand, x0, T, m, gamma_c, h, test_set, cost_per_iter, reps=reps, out_path=out_path)

T = budget
spider_szo_result = test_optimizer(target, "spider_szo", spider_szo, x0, T, m, gamma_c, h, test_set, None, reps=reps, out_path=out_path)

T = budget
zo_spider_coord_result = test_optimizer(target, "zo_spider_coord", zo_spider_coord, x0, T, m, gamma_c, h, test_set, None, reps=reps, out_path=out_path)

cost_per_iter = (l + 1)
T = budget // cost_per_iter
sszd_result = test_optimizer(target, "sszd", sszd_opt, x0, T, None, gamma, h, test_set, cost_per_iter, reps=reps, out_path=out_path)

cost_per_iter = (l + 1)
T = budget // cost_per_iter
gauss_result = test_optimizer(target, "gauss_opt", gauss_opt, x0, T, None, gamma, h, test_set, cost_per_iter, reps=reps, out_path=out_path)

cost_per_iter = (l + 1)
T = budget // cost_per_iter
sph_result = test_optimizer(target, "sph_opt", sph_opt, x0, T, None, gamma, h, test_set, cost_per_iter, reps=reps, out_path=out_path)
