import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BCELoss

import torchvision
from svrz.utils import TargetFunction

from torch.utils.data import Dataset


class LeastSquares(TargetFunction):
    
    def __init__(self, 
                 data : Dataset,
                 seed : int = 12131415,
                 ):
        super().__init__(data.n, seed)
        self.data = data
        
    def __call__(self, x, z = None, elem_wise = False):
        if z is None:
            return (1 / x.shape[1]) * (torch.matmul(x, self.data.A.T) - self.data.y).norm(p=2, dim=1, keepdim=True).square_()

        norm = 1.0 / z.shape[0] if len(z.shape) > 0 else 1.0
        if elem_wise:
            return norm * ( torch.einsum('bd,bd->b', self.data.A[z, :], x)  - self.data.y[z]).view(z.shape[0], 1).norm(p = 2, dim=1, keepdim=True).square_()
        return  norm * ( torch.matmul(x, self.data.A[z, :].T)  - self.data.y[z]).norm(p = 2, dim = 1, keepdim=True).square_()


class BBClassification(TargetFunction):
    
    def __init__(self, dataset, lam = 1e-3,seed: int = 12131415):
        super().__init__(dataset.n, seed)
        self.dataset = dataset
        self.lam = lam
        self.loss = BCELoss(reduction='none')
        
    def __call__(self, w, z=None, elem_wise = False):
        if z is None:
            return self.loss(torch.sigmoid(torch.matmul(w, self.dataset.X.T)), self.dataset.y.repeat(w.shape[0]).view(w.shape[0], -1)).mean(dim=1, keepdim=True) 
        if not elem_wise:
            return self.loss(torch.sigmoid(torch.matmul(w, self.dataset.X.T[:, z])), self.dataset.y[z].repeat(w.shape[0]).view(w.shape[0], -1)).mean(dim=1, keepdim=True)
        return self.loss(torch.einsum('bd,bd->b', self.dataset.X[z, :], w).sigmoid(), self.dataset.y[z]).view(z.shape[0], 1).mean(dim=1, keepdim=True)


class UniversalPerturbation(TargetFunction):
    
    def __init__(self, 
                 net,
                 data : Dataset, 
                 device : str = 'cpu',
                 dtype : torch.dtype = torch.float32,
                 batch_size : int = 128,
                 model_path = "./model/lenet_mnist_model.pth.pt",
                 seed: int = 12131415):
        super().__init__(n=data.n, seed=seed)
        self.data = data
        self.device =device
        self.batch_size = batch_size
        self.normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))
        self.net = self.net
        self.net.eval()
        
        
    # restores the tensors to their original scale
    def denorm(self, batch, mean=[0.1307], std=[0.3081]):

        if isinstance(mean, list):
            mean = torch.tensor(mean).to(self.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(self.device)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        
    def _apply_perturbation(self, images, perturbations):
        perturbations = perturbations.view(perturbations.shape[0], 1, 28, 28)
        #transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        return (self.denorm(images) + perturbations).clip(0, 1)
        
    def __call__(self, x, z=None, elem_wise = False):
        if z is None:
            losses = torch.zeros((x.shape[0], ))
            for i in range(self.data.X.shape[0] // self.batch_size):
                X = self.data.X[i*self.batch_size : (i + 1) * self.batch_size, :, :, :].view(self.batch_size, 1, 28, 28)
                y = self.data.y[i*self.batch_size : (i + 1) * self.batch_size]
                adv_imgs = self._apply_perturbation(X, x)
                adv_probs = self.net(adv_imgs)
                loss = F.nll_loss(adv_probs, y.to(dtype=torch.int64), reduction='sum')
                l2 = torch.norm(X - adv_imgs, p=2, dim=(1, 2, 3)).square().sum() / torch.norm(X , p=2).square().sum()
#                print(adv_imgs.shape, adv_probs.shape, x.shape, loss.shape, l2.shape)
                losses += -loss + l2
           #     print(adv_probs.max(1)[1], y)
            losses /= self.data.X.shape[0]
            return losses

        losses = torch.zeros((x.shape[0], ))
        for i in range(z.shape[0]):
            
            imgs, labels = self.data.X[z[i], :, :, :], self.data.y[z[i]]
            adv_imgs = self._apply_perturbation(imgs, x)
            labels = torch.tile(labels, (adv_imgs.shape[0], ))
            adv_probs = self.net(adv_imgs)
            loss = F.nll_loss(adv_probs, labels.to(dtype=torch.int64), reduction='none')
            l2 = torch.norm(imgs - adv_imgs,p=2,dim=(1,2,3)).square() / torch.norm(imgs,p=2).square()
           # print(adv_imgs.shape, adv_probs.shape, loss.shape, l2.shape)
            losses += -loss + l2
        losses /= z.shape[0]
        print(losses)
        #print(losses)
        return losses.view(-1, 1)
        # print("LOSS: ", loss, x.shape)
        # return F.nll_loss(adv_probs, labels.to(dtype=torch.int64))
