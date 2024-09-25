import sys
import torch
import numpy as np 

sys.path.append("../")
from datasets import RealDataset, load_libsvm_data
from utils import train_test_split, standardize



def test_optimizer(target, optimizer, x0, T, m, gamma, h, test_set, cost_per_iter = None, reps = 10):
    values, times = [], []
    te_error = []
    for _ in range(reps):
        if m is None:
            result = optimizer.optimize(target, x0 = x0, T = T, gamma = gamma, h = h)
        else:
            result = optimizer.optimize(target, x0 = x0, T = T, m = m, gamma = gamma, h = h)
        values.append(result['f_values'])
        times.append(result['it_times'])
        w = result['x']
        err = ((torch.sigmoid(torch.matmul(w, test_set.X.T)) > 0.5).to(torch.int64) != test_set.y).sum() / test_set.y.shape[0]
        te_error.append(err.item())

    print("-"*44)
    values = np.array(values).reshape(reps, -1)
    times = np.array(times).reshape(reps, -1).cumsum(1)
    ris = {
        'values' : (values.mean(0), values.std(0)),
        'times' : (times.mean(0), times.std(0)),
        'test_error' : (np.mean(te_error), np.std(te_error))
    }
    if cost_per_iter is None:
        cost_per_iter = result['l_values']

    ris['cost_per_iter'] = cost_per_iter
    return ris
            


def get_dataset(name, dtype = torch.float64, device = 'cpu', generator = None):
    if name == 'ijcnn1':
        X_tr, y_tr = load_libsvm_data(datapath="/data/mrando/ijcnn1/ijcnn1.tr", dtype=dtype, device=device)
        X_te, y_te = load_libsvm_data(datapath="/data/mrando/ijcnn1/ijcnn1.t", dtype=dtype, device=device)
    elif name == 'phishing':
        if generator is None:
            generator = torch.Generator(device=device)
        X, y = load_libsvm_data(datapath="/data/mrando/phishing/phishing", dtype=dtype, device=device)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, training_fraction=0.8, generator=generator)
    elif name == 'mushrooms':
        X, y = load_libsvm_data(datapath="/data/mrando/mushrooms/mushrooms", dtype=dtype, device=device)
        y[y==1] = -1.0
        y[y==2] = 1.0        
        if generator is None:
            generator = torch.Generator(device=device)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, training_fraction=0.8, generator=generator)
    else:
        raise ValueError("Dataset unknown! name must be ijcnn1, phishing or w8a")
    training_set = RealDataset(*standardize(X_tr, y_tr))
    test_set     = RealDataset(*standardize(X_te, y_te))
    return training_set, test_set
