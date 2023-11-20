import numpy as np

from gzvr import SSZD, SVRZ
from gzvr.directions import QRDirections, SphericalDirections
from targets import LinearRegression, LogisticRegression

import matplotlib.pyplot as plt

d, n = 50, 1000
seed = 12131415
fun = LinearRegression(d=d,n=n)


directions = [5, 10, 25, 50]
m = 1
T = 150

values = {}
L = max(np.linalg.eigvalsh(fun.A.T @ fun.A ))
budget = 10000
reps = 10
for l in directions:
    values[l] = []
    for _ in range(reps):
        x = np.full((d), 1.0)
        gamma = lambda k, x, fx  :  0.01 * (l/d) * (1 / np.sqrt(k + 1))# * (l/d)
        h = lambda k : 1e-10/np.sqrt(k) 

        P = QRDirections(d = d, l=l, seed=seed)
    #    opt = SVRZ(f = fun, n = n, P = P, P_full = P, seed=seed)
        opt = SSZD(f = fun, n = n, P = P, seed=seed)
        opt_ris = opt.optimize(x.copy(), T = budget // l, gamma=gamma, h = h, verbose=False)
        print(len(opt_ris['f_values']), opt_ris['num_evals'])
        values[l].append([])
        for fx in opt_ris['f_values']:
            values[l][-1] += [fx for _ in range(l)]
    values[l] = np.array(values[l]).reshape(reps, -1)
    values[l] = np.mean(values[l], axis=0)
    
fig, ax = plt.subplots()

for l in directions:
    ax.plot(range(len(values[l])), values[l], '-', label='$\ell = {}$'.format(l))
ax.set_yscale('log')
#ax.set_xscale('log')
ax.legend()
fig.savefig("./l_comparison.pdf", bbox_inches='tight')
