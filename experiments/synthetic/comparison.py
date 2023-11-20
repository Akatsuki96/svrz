import numpy as np

from gzvr import SSZD, SVRZ, Spider_ZD
from gzvr.directions import QRDirections, SphericalDirections
from targets import LinearRegression, LogisticRegression

d, n = 10, 100
seed = 12131415
fun = LinearRegression(d=d,n=n)

x = np.full((d), 1000.0)

P = QRDirections(d = d, l=d//2, seed=seed)
P1 = QRDirections(d =d , l = 2, seed=seed)
P2 = QRDirections(d = d, l=d//2, seed=seed)
opt = SSZD(f = fun, n = n, P = P, seed=seed)
opt2 = SVRZ(f = fun, n = n, P = P, seed=seed)
opt3 = SVRZ(f = fun, n = n, P = P1, P_full=P, seed=seed)
m = 25
T = 150
gamma = lambda k, x, fx  : 0.001 * (1/np.sqrt(k))
gamma2 = lambda k, x, fx : 0.01
h = lambda k : 1e-10/k 
svrz_ris = opt3.optimize(x.copy(), T = T, m=m, gamma=gamma2, h=h, verbose=True)
sszd_ris = opt.optimize(x.copy(), T = svrz_ris['num_evals'] // (d + 1), gamma=gamma, h = h, verbose=False)

print(svrz_ris['num_evals'], sszd_ris['num_evals'])

print("SVRZ1: {}\tSSZD: {}".format(svrz_ris['f_values'][-1], sszd_ris['f_values'][-1]))

import matplotlib.pyplot as plt

fig, ax =plt.subplots()

values = []
for (i, fx) in enumerate(svrz_ris['f_values']):
    values += [fx for _ in range(svrz_ris['lst_evals'][i])]

ax.plot(range(len(values)), values ,'-', label='svrz')
values = []
for (i, fx) in enumerate(sszd_ris['f_values']):
    values += [fx for _ in range(sszd_ris['lst_evals'][i])]

ax.plot(range(len(values)), values ,'-', label='sszd')
ax.legend()
ax.set_yscale('log')
fig.savefig("./test.pdf", bbox_inches='tight')
