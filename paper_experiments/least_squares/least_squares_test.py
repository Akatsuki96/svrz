import sys
import torch

from svrz.optimizers import OPSVRZ
from svrz.directions import QRDirections
from svrz.prox import SoftThreshold

sys.path.append("../")

from other_methods.rspgf import RSPGF
from other_methods.zo_pspider import ZOPSpider
from other_methods.zo_psvrg import ZOPSVRG

from targets import LeastSquares
from datasets import SyntheticDataset
from utils import get_optimizer, get_cost_per_iter



d = 50

opt_name = sys.argv[1]
L = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
mu = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

m = 25#50# int(sys.argv[4])

l = 10



dtype = torch.float64
device = 'cuda'
x_start = torch.zeros((d, ), dtype=dtype, device=device)

seed = 12131415
lam = 1e-5

data = SyntheticDataset(x_star=x_start,n = d, L = L, mu = mu, seed=seed)

target = LeastSquares(data=data, lam = lam, seed=seed)

prox = SoftThreshold(lam=lam)

x0 = torch.ones((1, d), dtype=dtype, device=device)
results = []
def callback(x, it_time, tau):
    print(f"[{tau}] F(x) = {target.full_target(x).flatten().item()}\ttime = {it_time}")

T = 30000

l_values = [1, 10, 25, 50]
m_values = [10, 25, 50] if opt_name != 'rspgf' else [1]
gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]

for gamma in gammas:
    cost_per_iter = get_cost_per_iter("opsvrz", d = d, l = l, m = m, n = d, b= 1)
    opt = OPSVRZ(P=QRDirections(d = d, l = l, seed = seed, device = device, dtype = dtype), batch_size=1, prox=prox, seed = seed)
    ris = opt.optimize(target, x0 = x0, T = T // cost_per_iter, m = m, gamma = gamma, h = lambda k : 1e-7, callback=callback)
    results.append(target.full_target(ris).flatten().item())
    print("----------------------------------------------")
# cost_per_iter = get_cost_per_iter("zo_psvrg_sph", d = d, l = l, m = m, n = d, b= 1)
# opt = ZOPSVRG(d = d, l=l, dir_type='spherical', dtype=dtype, device=device, prox=prox, seed = seed)
# ris =opt.optimize(target, x0 = x0, T = T // cost_per_iter, m = m, gamma = 0.01, h = lambda k : 1e-7, callback=callback)
# results.append(target.full_target(ris).flatten().item())
# print("----------------------------------------------")
# opt = RSPGF(d = d, l= l, prox = prox, seed = seed, dtype = dtype, device = device)
# cost_per_iter = get_cost_per_iter("rspgf", d = d, l = l, m = m, n = d, b= 1)
# ris = opt.optimize(target, x0 = x0, T = T // cost_per_iter, gamma = 0.01, h = lambda k : 1e-7, callback=callback)
# results.append(target.full_target(ris).flatten().item())


print(results)