import sys
import torch

from lenet import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


from utils import Attack

use_cuda = True

pretrained_model = "./data/lenet_mnist_model.pth.pt"

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=False)

torch.manual_seed(42)
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# Initialize the network
model = LeNet(device=device).to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

target = Attack(model, test_loader, lam=0.1, max_img=10, device=device)

v = torch.rand(target.data_shape, device=device)

#x_adv = target.generate_adv_example(target.denorm(target.X[0]), v.reshape(target.data_shape))
# print(x_adv)#target.generate_adv_example(target.X[0], v.reshape(target.data_shape)))

# print(model(transforms.Normalize((0.1307,), (0.3081,))(x_adv)))


from ssvr.optimizer import SZD, SVRZD, SARAH_ZD
from ssvr.directions import QRDirections

P = QRDirections(d = v.reshape(-1).shape[0], l = 5, device=device, dtype=torch.float32)
#opt = SARAH_ZD(target, n = target.n,  P = P)
opt = SZD(target, n = target.n,  P = P)

T = 500
#m = 10
#torch.set_num_threads(1)
from math import sqrt
opt.optimize(v.reshape(1, -1), T = T,gamma= lambda t,x,f : 1.0/sqrt(t + 1), h = lambda t : 1e-5, verbose=True)

# for data, target in test_loader:
#     data, target = data.to(device), target.to(device)
#     output = model(data)
#     init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#     if init_pred != target:
#         continue
#     loss = F.nll_loss(output, target)
#     print("[--] loss = {}".format(loss))

