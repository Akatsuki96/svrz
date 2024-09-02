import torch

from torch.utils.data import random_split


def train_test_split(X, y, training_fraction = 0.8, generator=torch.Generator()):
    tr_ind, te_ind = random_split(range(X.shape[0]), [training_fraction, 1 - training_fraction], generator=generator)
    return X[tr_ind, :], X[te_ind, :], y[tr_ind], y[te_ind]

def standardize(X, y):
    X_new = (X - X.mean())/X.std()
    y_new = y.clone()
    y_new[y_new < 0] = 0
    return X_new, y_new