import sys
import torch
import torchvision
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from math import floor, sqrt

from torch.utils.data import SubsetRandomSampler

import torch.nn.functional as F

from svrz.optimizers import SSZD, OSVRZ
from svrz.directions import QRDirections
import matplotlib.pyplot as plt 

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
        im_arct =  torch.arctanh(X.view(-1, 28 * 28) )

        #new_elem = (((im_at + v).tanh() + 1 )/2)
        if elem_wise:
            return ((X.view(-1, 28 * 28) + w).tanh() + 1) / 2
        return self.normalize(((( im_arct.repeat(w.shape[0], 1) + w.to(self.device).repeat_interleave(X.shape[0], dim=0)).tanh() + 1) / 2).view(-1, 1, 28, 28) )
    


    def __call__(self, w, z=None, elem_wise = False):
        if z is None:
            X = self.X
            y = self.y
        else:
            X = self.X[z, :].view(-1, 1, 28, 28)
            y = self.y[z]
#        print(net(v.view(-1, 1, 28, 28)))
        with torch.no_grad():
            results = torch.zeros((w.shape[0], 1), dtype=self.dtype, device=self.device)
            num_batches = w.shape[0] // batch_size + w.shape[0] % batch_size
            # print(batch_size, w.shape[0], num_batches)
            # print(w[:batch_size])
            # exit()
            for i in range(w.shape[0]):
#                print(i)
                v = self._apply_perturbation(X, w[i].view(1, -1), elem_wise=False).view(-1, 28 * 28)
                preds = F.softmax(self.net(v.view(-1, 1, 28, 28)), dim = 1)
                mask = torch.ones_like(preds, dtype=torch.bool, device=device)
                mask[torch.arange(mask.shape[0]), y.view(-1)] = False
                p1 = preds.gather(1, torch.tensor([self.label], dtype=torch.int64, device=device).repeat(preds.shape[0]).view(-1, 1)).view(v.shape[0], -1)#.log()
                p2 = preds[mask].view(mask.shape[0], -1).max(1)[0].view(-1, 1)#.log() # n x 1
                p2[p2 < 1e-10] = 1e-10
                reg = (v - X.view(-1, 28 * 28)).norm(2, keepdim=True).square()     
#                print(reg)
                results[i] = torch.maximum(p1.log() -  p2.log() , torch.zeros_like(p1, device=device)).mean(dim=0,keepdim=True) + 0.1 * reg#.mean()
            return results

        
def get_data_by_label(data_set, label, dtype = torch.float32, device='cpu'):
    indices = torch.argwhere(data_set.targets == label)[:5]
    X = data_set.data[indices, :, :].view(-1, 1, 28, 28).to(dtype).to(device) / 255
    #X = X - 0.5 / 0.5
    return X, data_set.targets[indices].flatten()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
batch_size=10
seed = 12131415

training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)


device = 'cuda'
dtype = torch.float64
net = MNISTNet(device=device).to(dtype)
net.load_state_dict(torch.load("./models/mnist/student", map_location=device))
net.eval()
label = 4
print("[--] Network loaded!")
X, Y = get_data_by_label(training_set, label=label, dtype=dtype, device=device)
print("[--] Data loaded!")

target = UniversalPerturbation(X, Y, net, label=label, seed = seed, batch_size=batch_size, dtype=dtype, device=device)
normalize = transforms.Normalize((0.5,), (0.5,))

d = 28*28
l = 50#d // 2

gamma = lambda k : 0.001 / sqrt(k + 1)#000001  * (1/ sqrt(k + 1)) * (5/d)
h = lambda k : 1e-5# / (k + 1)

T = 101



opt = OSVRZ(P = QRDirections(d = d, l = l, seed = seed, device=device, dtype=dtype), batch_size=1, seed=seed)
#opt = SSZD(P = QRDirections(d = d, l = l, seed = seed, device=device, dtype=dtype),  seed=seed)
x0 = torch.full((1, d), 0.1, dtype=dtype, device=device)

#inds = torch.randint(0, d, (2, ), device=device)
#x0[inds] = 0.0


ris = opt.optimize(target, x0, T = 200, gamma=0.001, m = 30, h = h)
w = ris['x']
x = normalize((((X[4].view(-1, 28 * 28).arctanh() + w).tanh() + 1) /2).view(1, 1, 28, 28))

print(w)

fig, ax = plt.subplots()

ax.plot(range(len(ris['f_values'])), ris['f_values'],'-')
ax.set_yscale('log')
plt.savefig("./loss.png")



fig, ax = plt.subplots()

ax.imshow( X[4].view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
plt.savefig("./testimg_orig.png")

#for i in range(len(ris['x_iters'])):
w = ris['x']
for i in range(X.shape[0]):
    x = normalize((((X[i].clone().view(-1, 28 * 28).arctanh()+ w).tanh() + 1) /2).view(1, 1, 28, 28))
    #x = (((X[4].clone().view(-1, 28 * 28).arctanh()+ w).tanh() + 1) /2).view(1, 1, 28, 28)
    y = F.softmax(net(normalize(x)), dim=1)

    print("[--] Prediction of perturbed = {} [{}]".format(y.max(1)[1].cpu().item(), y.max(1)[0].cpu().item()))

    fig, ax = plt.subplots()

    ax.imshow( x.view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
    plt.savefig(f"./testimg_final_{i}.png")
    plt.close(fig)


fig, ax = plt.subplots()

ax.imshow( w.view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
plt.savefig("./uni_pert.png")









