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
    
    def __init__(self, X, y, net, label, lam = 0.5, seed: int = 12131415, batch_size = 128, dtype = torch.float32, device = 'cpu'):
        super().__init__(X.shape[0], seed)
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        self.X = X#self.normalize(X)
        self.y = y
        self.net = net
        self.lam = lam
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
        with torch.no_grad():
            results = torch.zeros((w.shape[0], 1), dtype=self.dtype, device=self.device)
            num_batches = w.shape[0] // self.batch_size + int(w.shape[0] % self.batch_size > 0) if w.shape[0] > self.batch_size else w.shape[0]
            for i in range(num_batches):
#                print(w.shape, i, num_batches)
                batch = w[i * self.batch_size : (i + 1) * self.batch_size].view(-1, 28 * 28)
                v = self._apply_perturbation(X, batch, elem_wise=False).view(-1, 28 * 28) # flatten perturbed images
                preds = F.softmax(self.net(v.view(-1, 1, 28, 28)), dim = 1)
                # print(preds.shape)
                # print(preds)
                mask = torch.ones_like(preds, dtype=torch.bool, device=self.device)
                mask[torch.arange(mask.shape[0]), y.repeat(batch.shape[0]).view(-1)] = False
                p1 = preds.gather(1, torch.tensor([self.label], dtype=torch.int64, device=self.device).repeat(preds.shape[0]).view(-1, 1))#.view(v.shape[0], -1)#.log()
                p1 = p1.view(batch.shape[0], -1)
                p2 = preds[mask].view(mask.shape[0], -1).max(1)[0].view(batch.shape[0], -1)#.view(-1, 1)#.log() # n x 1
                p2[p2 < 1e-10] = 1e-10
 #               print(p1.shape, p2.shape)
                diff = torch.maximum(p1.log() - p2.log(), torch.zeros_like(p1, device=self.device)).mean(dim = 1)
               # print(v)
#                exit()
                reg = (v - X.view(-1, 28 * 28).repeat(batch.shape[0], 1)).view(batch.shape[0], -1, 28 * 28).norm(2, dim=2).square().mean(dim=1)
                results[i * self.batch_size : (i + 1) * self.batch_size] = (diff + self.lam * reg).view(batch.shape[0], 1) #torch.maximum(diff  , torch.zeros_like(p1, device=device)).view(self.batch_size, ).mean(dim=0,keepdim=True) + self.lam * reg#.mean()
            return results

        
def get_data_by_label(data_set, label, dtype = torch.float32, device='cpu'):
    indices = torch.argwhere(data_set.targets == label)[:20]
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

target = UniversalPerturbation(X, Y, net, label=label, lam=0.1, seed = seed, batch_size=batch_size, dtype=dtype, device=device)
normalize = transforms.Normalize((0.5,), (0.5,))

d = 28*28
l = 50#d // 2

gamma = lambda k : 0.001 / sqrt(k + 1)#000001  * (1/ sqrt(k + 1)) * (5/d)
h = lambda k : 1e-5# / (k + 1)

T = 101



opt = OSVRZ(P = QRDirections(d = d, l = l, seed = seed, device=device, dtype=dtype), batch_size=1, seed=seed)
#opt = SSZD(P = QRDirections(d = d, l = l, seed = seed, device=device, dtype=dtype),  seed=seed)
x0 = torch.full((1, d), 0.4, dtype=dtype, device=device)

#inds = torch.randint(0, d, (2, ), device=device)
#x0[inds] = 0.0


ris = opt.optimize(target, x0, T = 200, gamma=0.0001, m = 50, h = h)
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

print("W ", w.shape)
for i in range(X.shape[0]):
    x = normalize((((X[i].clone().view(-1, 28 * 28).arctanh()+ w).tanh() + 1) /2).view(1, 1, 28, 28))
    #x = (((X[4].clone().view(-1, 28 * 28).arctanh()+ w).tanh() + 1) /2).view(1, 1, 28, 28)
    y = F.softmax(net(x), dim=1)

    print("[--] Prediction of perturbed = {} [{}]".format(y.max(1)[1].cpu().item(), y.max(1)[0].cpu().item()))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow( x.view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
    ax2.imshow( X[i].view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
    fig.tight_layout()
    plt.savefig(f"./testimg_final_{i}.png")
    plt.close(fig)


fig, ax = plt.subplots()

ax.imshow( w.view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
plt.savefig("./uni_pert.png")









