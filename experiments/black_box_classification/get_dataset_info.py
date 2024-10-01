import os
import sys
import torch
import numpy as np

from bbclass_utils import get_dataset


dataset_names = ['ijcnn1', 'phishing', 'mushrooms']


for name in dataset_names:
    tr_dataset, te_dataset = get_dataset(name)
    print(f"[++] Name = {name}")
    print("\t[--] Training data shape = {}\tTest data shape = {}".format(tr_dataset.X.shape, te_dataset.X.shape))
    