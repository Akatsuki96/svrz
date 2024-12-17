import torch

from torch.utils.data import random_split

from svrz.optimizers import OPSVRZ
from svrz.directions import QRDirections
from svrz.prox import ProxOperator

from other_methods.rspgf import RSPGF
from other_methods.zo_psvrg import ZOPSVRG
from other_methods.zo_pspider import ZOPSpider

def train_test_split(X, y, training_fraction = 0.8, generator=torch.Generator()):
    tr_ind, te_ind = random_split(range(X.shape[0]), [training_fraction, 1 - training_fraction], generator=generator)
    return X[tr_ind, :], X[te_ind, :], y[tr_ind], y[te_ind]

def standardize(X, y):
    X_new = (X - X.mean())/X.std()
    y_new = y.clone()
    return X_new, y_new


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