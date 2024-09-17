import torch

from torch.utils.data import random_split

from svrz.optimizers import *
from svrz.directions import *


def train_test_split(X, y, training_fraction = 0.8, generator=torch.Generator()):
    tr_ind, te_ind = random_split(range(X.shape[0]), [training_fraction, 1 - training_fraction], generator=generator)
    return X[tr_ind, :], X[te_ind, :], y[tr_ind], y[te_ind]

def standardize(X, y):
    X_new = (X - X.mean())/X.std()
    y_new = y.clone()
    y_new[y_new < 0] = 0
    return X_new, y_new


def get_optimizer(name, d, l, dtype = torch.float64, device = 'cpu', seed = 1231415):    
    if name == 'sszd':
        return SSZD(P = QRDirections(d=d, l = l, seed = seed, device=device, dtype=dtype), nrm_const=d / l, seed=seed)
    elif name == 'gauss_fd':
        return SSZD(P = GaussianDirections(d=d, l = l, seed = seed, device=device, dtype=dtype), nrm_const=1 /l, seed=seed)
    elif name == 'sph_fd':
        return SSZD(P = SphericalDirections(d=d, l = l, seed = seed, device=device, dtype=dtype), nrm_const= d / l, seed=seed)
    elif name == 'zosvrg_ave':
        return ZOSVRG(d = d, l = l, dtype = dtype, device = device, seed=seed, estimator='ave')
    elif name == 'zosvrg_coord':
        return ZOSVRG(d = d, l = l, dtype = dtype, device = device, seed=seed, estimator='coord')
    elif name == 'szvr_g':
        return SZVR_G(d = d, l = l, dtype = dtype, device = device, seed = seed)
    elif name == 'zosvrg_cr':
        return ZOSVRG_CoordRand(d = d, l = l, dtype = dtype, device = device, seed = seed)
    elif name == 'zospider_szo':
        return SpiderSZO(d = d, l = l, dtype =dtype, device =device, seed = seed)
    elif name == 'zospider_coord':
        return ZOSpiderCoord(d = d, batch_size=1, dtype =dtype, device =device, seed = seed)
    elif name == 'osvrz':
        return OSVRZ(P = QRDirections(d = d, l = l, seed = seed, device = device, dtype = dtype), batch_size=1, seed=seed)
    raise ValueError(f"Algorithm {name} is unknown!")


def get_cost_per_iter(name, d, l, m, n):    
    if name == 'sszd':
        return l + 1
    elif name == 'gauss_fd':
        return l + 1
    elif name == 'sph_fd':
        return l + 1
    elif name == 'zosvrg_ave':
        return (2 * (l + 1) * m + n * (l + 1))
    elif name == 'zosvrg_coord':
        return (4 * d * m + 2 * n * d)
    elif name == 'szvr_g':
        return (4 * m + n * (l + 1))
    elif name == 'zosvrg_cr':
        return (4 * l * m + 2 * n * d)
    elif name == 'zospider_szo':
        return None 
    elif name == 'zospider_coord':
        return None 
    elif name == 'osvrz':
        return (2 * (l + 1)  * m + n * (d + 1) )
    raise ValueError(f"Algorithm {name} is unknown!")