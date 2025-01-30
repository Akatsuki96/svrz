import torch
import numpy as np

from torch.utils.data import random_split

from svrz.optimizers import OPSVRZ
from svrz.directions import QRDirections
from svrz.prox import ProxOperator

from other_methods.rspgf import RSPGF
from other_methods.zo_psvrg import ZOPSVRG
from other_methods.zo_pspider import ZOPSpider

from datasets import RealDataset, load_libsvm_data

def train_test_split(X, y, training_fraction = 0.8, generator=torch.Generator()):
    tr_ind, te_ind = random_split(range(X.shape[0]), [training_fraction, 1 - training_fraction], generator=generator)
    return X[tr_ind, :], X[te_ind, :], y[tr_ind], y[te_ind]

def standardize(X, y):
    X_new = (X - X.mean())/X.std()
    y_new = y.clone()
    return X_new, y_new

class ResultCallback:
    
    def __init__(self, target, T, elem_wise = False, reps = 10):
        self.target = target
        self.elem_wise = elem_wise
        self.values = np.zeros((reps, T + 1))
        self.it_times = np.zeros((reps, T + 1))
        self.rep_counter = 0
        
    def __call__(self, x, it_time, tau):
        self.values[self.rep_counter, tau] =  self.target.full_target(x, elem_wise=self.elem_wise)
        self.it_times[self.rep_counter, tau] = it_time
        print(tau, self.values[self.rep_counter, tau])
        
    def contains_nan_values(self):
        return np.any(self.values != self.values)
        
    def go_next(self):
        self.rep_counter += 1


        
def test_optimizer(target, optimizer, x0, T, m, gamma, h, reps = 10):
    cb = ResultCallback(target, T, elem_wise=False, reps=reps)
    for _ in range(reps):
        _ = optimizer.optimize(target, x0 = x0, T = T, m = m, gamma = gamma, h = h, callback=cb)
        cb.go_next()
    values, times = cb.values, cb.it_times.cumsum(1)
    return {
        'values' : (values.mean(0), values.std(0)),
        'times' : (times.mean(0), times.std(0))
    }





def get_dataset(name, data_path, dtype = torch.float64, device = 'cpu', generator = None):
    if name == 'ijcnn1':
        X, y = load_libsvm_data(datapath=f"{data_path}/ijcnn1/ijcnn1.tr", dtype=dtype, device=device)
        y[y < 0] = 0.0
    elif name in ['phishing', 'australian', 'german.numer', 'splice']:#== 'phishing' or name == 'australian' or name == 'a1a' or name:
        X, y = load_libsvm_data(datapath=f"{data_path}/{name}/{name}", dtype=dtype, device=device)
        y[y < 0] = 0.0
    elif name == 'mushrooms':
        X, y = load_libsvm_data(datapath=f"{data_path}/mushrooms/mushrooms", dtype=dtype, device=device)
        y[y==1] = 0.0
        y[y==2] = 1.0        
    return RealDataset(*standardize(X, y))


def get_optimizer(name, d, l, prox : ProxOperator, b = 1, dtype = torch.float64, device = 'cpu', seed = 1231415):    
    if name == 'rspgf':
        return RSPGF(d=d, l=l, prox=prox, seed=seed, dtype=dtype, device=device) 
    elif name == 'zo_pspider':
        return ZOPSpider(d = d, l = l, prox=prox, dtype =dtype, device =device, seed = seed, estimator='randsge')
    elif name == 'zo_pspider_coord':
        return ZOPSpider(d = d, l=d, prox=prox, dtype =dtype, device =device, seed = seed, estimator='coosge')
    elif name == 'zo_psvrg_sph':
        return ZOPSVRG(d = d, l = l, dir_type='spherical', prox = prox, dtype = dtype, device = device, seed = seed)
    elif name == 'zo_psvrg_gaus':
        return ZOPSVRG(d = d, l = l, dir_type='gaussian', prox = prox, dtype = dtype, device = device, seed = seed)
    elif name == 'zo_psvrg_coord':
        return ZOPSVRG(d = d, l = d, dir_type='coordinate', prox = prox, dtype = dtype, device = device, seed = seed)
    elif name == 'opsvrz':
        return OPSVRZ(P = QRDirections(d = d, l = l, seed = seed, device = device, dtype = dtype), batch_size=b, prox=prox, seed=seed)
    raise ValueError(f"Algorithm {name} is unknown!")


def get_cost_per_iter(name, d, l, m, n, b):    
    if name == 'rspgf':
        return l + 1
    elif name == 'zo_pspider':
        return 2 * d * n + 4 * m * l 
    elif name == 'zo_pspider_coord':
        return 2 * d * n + 4 * m * d 
    elif name == 'zo_psvrg_sph' or name == 'zo_psvrg_gaus':
        return 2 * d * n + 4 * m * l
    elif name == 'zo_psvrg_coord':
        return 2 * d * n + 4 * m * d
    elif name == 'opsvrz':
        return n * (d + 1) + 2 * b * m * (l + 1)
    raise ValueError(f"Algorithm {name} is unknown!")