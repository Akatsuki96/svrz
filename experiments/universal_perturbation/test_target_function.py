import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
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
            return ((im_arct + w).tanh() + 1) / 2
#        return self.normalize(((( im_arct.repeat(w.shape[0], 1) + w.to(self.device).repeat_interleave(X.shape[0], dim=0)).tanh() + 1) / 2).view(-1, 1, 28, 28) )
        return self.normalize(((( im_arct.repeat(w.shape[0], 1).view(w.shape[0], -1, 28 * 28) + w.view(w.shape[0], 1, 28*28)).tanh() + 1) / 2).view(-1, 1, 28, 28) )
   


    def __call__(self, w, z=None, elem_wise = False):
        if z is None:
            X = self.X
            y = self.y
        else:
            X = self.X[z, :].view(-1, 1, 28, 28)
            y = self.y[z]
        with torch.no_grad():
            results = torch.zeros((w.shape[0], 1), dtype=self.dtype, device=self.device)
            num_batches = w.shape[0] // self.batch_size + int(w.shape[0] % self.batch_size > 0) #if w.shape[0] > self.batch_size else w.shape[0]
            for i in range(num_batches):
   #             print(w.shape, i, num_batches)
                batch = w[i * self.batch_size : (i + 1) * self.batch_size].view(-1, 28 * 28)
                v = self._apply_perturbation(X, batch, elem_wise=False).view(-1, 28 * 28) # flatten perturbed images
  #              print(v.shape)
                preds = F.softmax(self.net(v.view(-1, 1, 28, 28)), dim = 1)
 #               print(preds.shape)
                mask = torch.ones_like(preds, dtype=torch.bool, device=self.device)
                mask[torch.arange(mask.shape[0]), y.repeat(batch.shape[0]).view(-1)] = False
                p1 = preds.gather(1, torch.tensor([self.label], dtype=torch.int64, device=self.device).repeat(preds.shape[0]).view(-1, 1))#.view(v.shape[0], -1)#.log()
                p1 = p1.view(batch.shape[0], -1)
                p2 = preds[mask].view(mask.shape[0], -1).max(1)[0].view(batch.shape[0], -1)#.view(-1, 1)#.log() # n x 1
                p2[p2 < 1e-50] = 1e-50
                diff = torch.maximum(p1.log() - p2.log(), torch.zeros_like(p1, device=self.device)).mean(dim = 1)
                reg = (v.view(batch.shape[0], -1, 28 * 28) - X.view(-1, 28 * 28).repeat(batch.shape[0], 1).view(batch.shape[0], -1, 28 * 28)).norm(2, dim=2).square().mean(dim=1)
#                print(diff, reg, diff + self.lam * reg)
                results[i * self.batch_size : (i + 1) * self.batch_size] = (diff + self.lam * reg).view(batch.shape[0], 1) #torch.maximum(diff  , torch.zeros_like(p1, device=device)).view(self.batch_size, ).mean(dim=0,keepdim=True) + self.lam * reg#.mean()
            return results

normalize = transforms.Normalize((0.5,), (0.5,))
        
def get_data_by_label(net, data_set, label, dtype = torch.float32, device='cpu', num_tr_examples = 50):
    indices = torch.argwhere(data_set.targets == label)
    X = data_set.data[indices, :, :].view(-1, 1, 28, 28).to(dtype).to(device) / 255
    
    X_new_tr, X_new_te = [], []
    Y_new_tr, Y_new_te = [], []
    for i in range(X.shape[0]):
        pred  = net(X[i].view(1, 1, 28, 28)).max(1)[1].item()
        if pred != label:
            continue
        if len(X_new_tr) < num_tr_examples:
            X_new_tr.append(X[i].tolist())
            Y_new_tr.append(label)        
        else:
            X_new_te.append(X[i].tolist())
            Y_new_te.append(label)        
             
        
    X_new_tr = torch.tensor(X_new_tr, dtype=dtype, device=device).view(-1, 1, 28, 28)
    Y_new_tr = torch.tensor(Y_new_tr, dtype=torch.int64, device=device).view(-1)
    X_new_te = torch.tensor(X_new_te, dtype=dtype, device=device).view(-1, 1, 28, 28)
    Y_new_te = torch.tensor(Y_new_te, dtype=torch.int64, device=device).view(-1)
    
    #X = X - 0.5 / 0.5
    return X_new_tr, Y_new_tr, X_new_te, Y_new_te #data_set.targets[indices].flatten()

def get_evasion_rate(net, X, Y, w):
    evasion_rate = []
    confidence = []
    for i in range(X.shape[0]):
        x = normalize((((X[i].clone().view(-1, 28 * 28).arctanh()+ w).tanh() + 1) /2).view(1, 1, 28, 28))
        y = F.softmax(net(x), dim=1).max(1)
        label, conf = y[1], y[0]
        evaded = int(label.cpu().item() != Y[i].cpu().item())
        evasion_rate.append(evaded)
        if evaded:
            confidence.append(conf.cpu().item())
    return np.mean(evasion_rate), np.mean(confidence) if len(confidence) > 0 else 0.0, np.std(confidence)  if len(confidence) > 0 else 0.0
        
        

def run_experiment(opt_name, net, opt, x0, T, m, gamma, h, reps=1):
    fvalues = []
    tr_evrate, te_evrate = [], []
    for i in range(reps):
        if m is not None:
            ris = opt.optimize(target, x0, T = T, gamma=gamma, m = m, h = h)
        else:
            ris = opt.optimize(target, x0, T = T, gamma=gamma, h = h)

        fvalues.append(ris['f_values'])
        ev_tr, _, _ = get_evasion_rate(net, X, Y, ris['x'])
        ev_te, _, _ = get_evasion_rate(net, X_te, Y_te, ris['x'])
        tr_evrate.append(ev_tr)
        te_evrate.append(ev_te)
    fvalues = np.array(fvalues).reshape(reps, -1)
    mu_values = fvalues.mean(axis=0)
    std_values = fvalues.std(axis=0)
    
    with open(f"{out_dir}/{opt_name}.log", 'w') as f:
        for i in range(fvalues.shape[0]):
            f.write("{},{}\n".format(mu_values[i], std_values[i]))
            
    mu_tr_ev = np.mean(tr_evrate) if len(tr_evrate) > 0 else 0.0
    mu_te_ev = np.mean(te_evrate) if len(te_evrate) > 0 else 0.0
    std_tr_ev = np.std(tr_evrate) if len(tr_evrate) > 0 else 0.0
    std_te_ev = np.std(te_evrate) if len(te_evrate) > 0 else 0.0
            
    with open(f"{out_dir}/{opt_name}_evasion.log", 'w') as f:
        for i in range(fvalues.shape[0]):
            f.write("{},{},{},{}\n".format(mu_tr_ev, std_tr_ev, mu_te_ev, std_te_ev))

    


transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
batch_size=50
seed = 12131415

training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)


device = 'cuda'
dtype = torch.float64
net = MNISTNet(device=device).to(dtype)
net.load_state_dict(torch.load("./models/mnist/student", map_location=device))
net.eval()
label = 9
num_tr_examples = 150
print("[--] Network loaded!")
X, Y, X_te, Y_te = get_data_by_label(net, training_set, label=label, dtype=dtype, device=device, num_tr_examples=num_tr_examples)
print("[--] Data loaded!")

out_dir = "/data/mrando/svrz_results/universal_perturbation"
img_dir = f"{out_dir}/samples"

os.makedirs(img_dir, exist_ok=True)

target = UniversalPerturbation(X, Y, net, label=label, lam=2.0, seed = seed, batch_size=batch_size, dtype=dtype, device=device)

d = 28*28
l = 10

gamma = lambda k : (l/ d) * (0.01 / sqrt(k + 1))#000001  * (1/ sqrt(k + 1)) * (5/d)
h = lambda k : 1.0


budget = 123132000

T = 500#0
m = 10

generator = torch.Generator(device=device).manual_seed(123132321)
opt_name = "osvrz"
opt = OSVRZ(P = QRDirections(d = d, l = l, seed = seed, device=device, dtype=dtype), batch_size=1, seed=seed)
sszd = SSZD(P = QRDirections(d = d, l = l, seed = seed, device=device, dtype=dtype),  seed=seed)
x0 = 0.001 * torch.randn((1, d), dtype=dtype, device=device, generator=generator)# torch.full((1, d), 0.01, dtype=dtype, device=device)


#(2 * (l + 1)  * m + d * (d + 1) ) * T

run_experiment('sszd', net, sszd, x0.clone(), budget // (l + 1), None, gamma, h, reps=1)

cost_per_iter = (2 * (l + 1)  * m + d * (d + 1) )
T = budget // cost_per_iter
ris = opt.optimize(target, x0, T = T, gamma=0.0031 * l/d, m = m, h = h)
w = ris['x']

torch.save(w, f"{out_dir}/osvrz_up")

print(w)

fig, ax = plt.subplots()
ax.set_title("Function values")
ax.plot(range(len(ris['f_values'])), ris['f_values'],'-')
ax.set_yscale('log')
ax.set_xlabel("iterations")
ax.set_ylabel("$F(x^\\tau)$")
plt.savefig(f"./loss.png")

evasion_rate = []
confidence = []
print("W ", w.shape)
for i in range(X.shape[0]):
    x = normalize((((X[i].clone().view(-1, 28 * 28).arctanh()+ w).tanh() + 1) /2).view(1, 1, 28, 28))
    #x = (((X[4].clone().view(-1, 28 * 28).arctanh()+ w).tanh() + 1) /2).view(1, 1, 28, 28)
    y = F.softmax(net(x), dim=1)
    y_real = F.softmax(net(normalize(X[i].clone().view(1, 1, 28 , 28))), dim=1)

    print("[{}/{}] Prediction real = {} [{}]\t perturbed = {} [{}]".format(i, X.shape[0], y_real.max(1)[1].cpu().item(), round(y_real.max(1)[0].cpu().item(), 2), y.max(1)[1].cpu().item(), round(y.max(1)[0].cpu().item(), 2)))
    evaded = y.max(1)[1].cpu().item() != label
    evasion_rate.append(evaded)
    if evaded and np.sum(evasion_rate) < 10:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.set_title("Real Example (classified as {} [{}])".format(y_real.max(1)[1].cpu().item(), round(y_real.max(1)[0].cpu().item(), 2)))
        ax2.set_title("Adversarial Example (classified as {} [{}])".format(y.max(1)[1].cpu().item(), round(y.max(1)[0].cpu().item(), 2)))
        ax1.imshow( X[i].view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
        ax2.imshow( x.view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
        fig.tight_layout()
        plt.savefig(f"./{opt_name}_final_{i}.png", bbox_inches='tight')
        plt.close(fig)
    if evaded:
        confidence.append(y.max(1)[0].cpu().item())
        
mu_tr = np.mean(confidence) if len(confidence) > 0 else 0.0
std_tr = np.std(confidence) if len(confidence) > 0 else 0.0

print("[TR] Evasion Rate = {} \t Confidence = {} +/- {}".format(np.mean(evasion_rate), mu_tr, std_tr))

erate_test, mu_conf_test, std_conf_test = get_evasion_rate(net, X_te, Y_te, w)
print("[TE ({})] Evasion Rate = {} \t Confidence = {} +/- {}".format(X_te.shape[0], erate_test, mu_conf_test, std_conf_test))

fig, ax = plt.subplots()

cax = ax.imshow( w.view(28, 28).cpu().detach().numpy(), cmap='gray')#.view(28, 28))
fig.colorbar(cax, ax=ax)
plt.savefig("./uni_pert.png")









