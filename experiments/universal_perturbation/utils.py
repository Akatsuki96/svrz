from typing import Any
import torch

import numpy as np

from math import log
from itertools import islice
from torchvision import datasets, transforms
class Attack:
    
    def __init__(self, model, data_loader, class_label=1, max_img = None,lam = 0.1, device = 'cpu') -> None:
        self.model = model
        self.data_loader = data_loader
        self.lam = lam
        self.max_img = max_img
        self.class_label = class_label
        self.device = device
        self.X, self.y = [], []
        self._initialize()
    @property
    def n(self):
        return len(self.X)

    @property
    def data_shape(self):
        return self.X[0].shape
            
    def _initialize(self):
        for x, y in self.data_loader:
            x.requires_grad = False
            out = self.model(x).max(1, keepdim=True)[1]
            img_shape = list(x.shape)
            if out.cpu() != y:
                continue
            if y.item() == self.class_label:
                self.X.append(x.tolist())
                self.y.append(y.tolist())
            if self.max_img is not None and len(self.X) == self.max_img:
                break
        img_shape[0] = len(self.X)
        self.X = torch.Tensor(self.X).reshape(img_shape).to(device=self.device)
        self.y = torch.Tensor(self.y).reshape((-1, 1)).to(device=self.device)
        
    def generate_adv_example(self, x, v):
        x_adv = 0.5*torch.tanh(2*torch.atanh(x) + v)#x + v# 0.5 * torch.tanh(torch.atanh(2*x) + v)
       # print(torch.tanh(torch.atanh(x) + v))
        return torch.clamp(x_adv, 0.0, 1.0)
    
    
    def denorm(self, batch, mean=[0.1307], std=[0.3081]):
        if isinstance(mean, list):
            mean = torch.tensor(mean, device=self.device)
        if isinstance(std, list):
            std = torch.tensor(std, device=self.device)
        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        
    def _loss(self, x, y, v):
        y = y.to(torch.int64).reshape(-1)
        x_adv = self.generate_adv_example(self.denorm(x), v)
        x_adv = transforms.Normalize((0.1307,), (0.3081,))(x_adv)
        output = self.model(x_adv)
        # print("XADV OUT", x_adv.shape, output.shape)    
        F_y = output[range(output.shape[0]), y].reshape(-1, 1)
        F_t = output#.clone()
        F_t[range(output.shape[0]), y] = -np.inf
        # print(F_y.shape, F_t.shape)
        # print(F_t.max(dim=1, keepdims=True))       
        adv_error = (F_y - F_t.max(dim=1, keepdims=True).values)
        distortion = (x_adv - x)
        adv_distortion = torch.linalg.norm(distortion.reshape(x.shape[0], -1), dim=1, keepdims=True)
        # print(adv_distortion.shape)       
        # print(torch.maximum(adv_error, torch.zeros(adv_error.shape, device=self.device)).shape)
        # exit()
        #print(adv_error.device, adv_distortion.device)
        loss = torch.maximum(adv_error, torch.zeros(adv_error.shape, device='cuda')) + self.lam * adv_distortion
       # print(loss.shape)
        return loss
    
    def __call__(self, v, z = None) -> Any:
        v = v.view((-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
        if z is None:
   #         print("X Y V", self.X.shape, self.y.shape, v.shape)
            return self._loss(self.X, self.y, v).mean(dim=0, keepdim=True)
        x, y = self.X[z, :, :, :], self.y[z]
    #    print("S: ", self.X.shape, self.y.shape)
        return self._loss(x, y, v)