import os
import numpy as np

from targets import LinearRegression
from ssvr.optimizer import SZD, SVRSZD
from ssvr.directions import QRDirections
import matplotlib.pyplot as plt


def write_result(out_file, trace, it_times):
    with open(f"{out_file}", 'w') as f:
        for i, j in zip(range(len(trace)), range(len(it_times))):
            f.write(f"{trace[i]},{it_times[j]}\n")



d = 5
l = d
n = 1000
seed = 121314
P = QRDirections(d=d,l=l,seed=seed)
f = LinearRegression(d = d, n=n, seed=seed)

sszd = SZD(fun = f, d = d, n =n, l = l, P = P, seed=seed)
ssvr = SVRSZD(fun =f, d =d, l=l, n=n, P = P, seed=seed)

gamma = lambda t : 0.001 * (l/d) * (1/np.sqrt(t + 1))

T1 = 100
m = 50
T2 = (T1 * (2 * m * (l + 1) + (l + 1) * n)) // (l + 1)#10100

x0 = np.full(shape=d, fill_value=1200.0)

ssvrz_result = ssvr.optimize(x0.copy(), m = m, T = T1, gamma = 0.001 * (l/d), h = lambda t : 1e-3/( (t + 1) ** 2), verbose=True)
sszd_result = sszd.optimize(x0.copy(), T = T2, gamma=gamma, verbose=True, h = lambda t : 1e-3/( (t + 1) ** 2))

sszd_values, sszd_times = [], []
svrz_values, svrz_times = [], []

szd_tm, svrz_tm = np.cumsum(sszd_result['it_times']), np.cumsum(ssvrz_result['it_times'])
for i in range(T2):
    sszd_values += [sszd_result['f_values'][i] for _ in range(l + 1)]
    sszd_times += [szd_tm[i] for _ in range(l + 1)]

for i in range(T1):
    svrz_values += [ssvrz_result['f_values'][i] for _ in range(2 * (l + 1) * m + (l + 1)*n)]
    svrz_times += [svrz_tm[i] for _ in range(2 * (l + 1) * m + (l + 1)*n)]

#print(len(svrz_values) // (l + 1))

print(len(svrz_values), len(sszd_values))
out_path = "./results/linear_regression"
os.makedirs(out_path, exist_ok=True)
write_result(f"{out_path}/sszd.log", sszd_values, sszd_times)
write_result(f"{out_path}/svrz.log", svrz_values, svrz_times)

        