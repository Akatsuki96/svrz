import sys
import torch
import torchvision
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from math import floor, sqrt

from torch.utils.data import SubsetRandomSampler

import torch.nn.functional as F

from svrz.optimizers import SSZD
from svrz.directions import QRDirections

sys.path.append("../")
from net import MNISTNet
from targets import TargetFunction


class UniversalPerturbation(TargetFunction):
    
    def __init__(self, X, y, net, label, seed: int = 12131415, batch_size = 128, dtype = torch.float32, device = 'cpu'):
        super().__init__(X.shape[0], seed)
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        self.X = X#self.normalize(X)
        self.y = y
        self.net = net
        self.label = label
        self.batch_size = batch_size
        self.device = device
        self.dtype=dtype
        self.net.eval()
        
    def _apply_perturbation(self, X, w, elem_wise = False):
        if elem_wise:
            return ((X.view(-1, 28 * 28) + w).tanh() + 1) / 2
#        return (( 1.9 * (X.view(-1, 28 * 28).repeat(w.shape[0], 1)).atan() + w.repeat_interleave(X.shape[0], dim=0)).tanh() + 1)/2
        return (( (X + 0.5).arctanh().view(-1, 28 * 28).repeat(w.shape[0], 1) + w.repeat_interleave(X.shape[0], dim=0)).tanh() -0.5)# / 2#((X.view(-1, 28 * 28).repeat(w.shape[0], 1) + w.repeat_interleave(X.shape[0], dim=0)).tanh() + 1) / 2
        #(X[0] + 0.5).view(1, -1).arctanh()
    def __call__(self, w, z=None, elem_wise = False):
        if z is None:
            X = self.X
            y = self.y
        else:
            X = self.X[z, :].view(-1, 1, 28, 28)
            y = self.y[z]
        v = self._apply_perturbation(X, w, elem_wise=elem_wise) # n x d
        with torch.no_grad():
            preds = F.softmax(self.net(v.view(-1, 1, 28, 28)), dim=1) # n x 10
    #        print(preds)
    #        exit()
            mask = torch.ones_like(preds, dtype=torch.bool) # n x 10
            # if elem_wise:
            #     mask[torch.arange(mask.shape[0]), y.view(-1)] = False
            #     p1 = preds.gather(1, y.view(-1, 1)).view(w.shape[0], -1).log()  # n x 1
            # else:
            #     mask[torch.arange(mask.shape[0]), y.repeat(w.shape[0]).view(-1)] = False                
            #     p1 = preds.gather(1, y.repeat(w.shape[0]).view(-1, 1)).view(w.shape[0], -1).log()  # n x 1
            p1 = preds.gather(1, torch.tensor([self.label], dtype=torch.int64).repeat(preds.shape[0]).view(-1, 1)).view(w.shape[0], -1).log()
            p2 = preds[mask].view(mask.shape[0], -1).max(1)[0].view(w.shape[0], -1).log() # n x 1
    #        print( preds.gather(1, torch.tensor([self.label], dtype=torch.int64).repeat(preds.shape[0]).view(-1, 1)).shape)
            reg = (v - X.view(-1, 28 * 28)).norm(2, keepdim=True).square()#.view(w.shape[0], -1).mean(dim=1, keepdim=True)
    #        print(preds.gather(1, torch.tensor([self.label], dtype=torch.int64).repeat(preds.shape[0]).view(-1, 1)).view(w.shape[0], -1))#.mean(1, keepdim=True))
            return torch.maximum(p1 - p2 , torch.zeros_like(p1)).mean(dim=1,keepdim=True) +  0.1* reg #torch.maximum(p1 - p2, torch.zeros_like(p1)).mean(dim=1,keepdim=True) #+ 0.1 * reg

        
def get_data_by_label(data_set, label, dtype = torch.float32, device='cpu'):
    indices = torch.argwhere(data_set.targets == label)[:1000]
    X = (data_set.data[indices, :, :].view(-1, 1, 28, 28).to(dtype).to(device) / 255) - 0.5
    #X = X - 0.5 / 0.5
    return X, data_set.targets[indices].flatten()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
batch_size=128
seed = 12131415

training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)


device = 'cpu'
dtype = torch.float64
net = MNISTNet(device=device).to(dtype)
net.load_state_dict(torch.load("./models/mnist/student", map_location=device))
net.eval()
label = 2
print("[--] Network loaded!")
X, Y = get_data_by_label(training_set, label=label, dtype=dtype, device=device)
print("[--] Data loaded!")

target = UniversalPerturbation(X, Y, net, label=label, seed = seed, dtype=dtype, device=device)

print(net(X[0].view(1, 1, 28, 28)).max(1)[1])

#print(1.99999 * X[0].atan())
#exit()

d = 28*28
l = d // 2

gamma = lambda k : 0.01 / sqrt(k + 1)#000001  * (1/ sqrt(k + 1)) * (5/d)
h = lambda k : 1e-3 / (k + 1)

T = 500


#print(X[0])
#exit()


opt = SSZD(P = QRDirections(d = d, l = 5, seed = seed, device=device, dtype=dtype),  nrm_const=d/2, seed=seed)
x0 = torch.full((1, d), 0.0, dtype=dtype, device=device)

v = (( (X[0] + 0.5).view(1, -1).arctanh() + x0).tanh() - 0.5 ) 

#print((v - X[0].view(-1, 28 * 28)).norm(2, keepdim=True).square())
#exit()
#x = X[0]
#x_norm = target.normalize(X[0])
#x_denorm = 2* x_norm  - 1#* 0.5

#print(x, x_denorm)
#exit()
ris = opt.optimize(target, x0, T = T, gamma=gamma, h = h)


print(ris['f_values'])


w = ris['x']

x = ((X[10].view(-1, 28 * 28) + w).tanh() + 1) /2


print(w)

import matplotlib.pyplot as plt 

fig, ax = plt.subplots()

ax.plot(range(len(ris['f_values'])), ris['f_values'],'-')
ax.set_yscale('log')
plt.savefig("./loss.png")



fig, ax = plt.subplots()

ax.imshow( X[10].view(28, 28).detach().numpy(), cmap='gray')#.view(28, 28))
plt.savefig("./testimg_orig.png")


fig, ax = plt.subplots()

ax.imshow( x.view(28, 28).detach().numpy(), cmap='gray')#.view(28, 28))
plt.savefig("./testimg.png")

print(net(x.view(1,1,28, 28)).max(1)[1], net(X[0].view(1,1,28, 28)).max(1)[1])


fig, ax = plt.subplots()

ax.imshow( w.view(28, 28).detach().numpy(), cmap='gray')#.view(28, 28))
plt.savefig("./uni_pert.png")









