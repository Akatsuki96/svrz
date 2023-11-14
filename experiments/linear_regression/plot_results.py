import numpy as np

import matplotlib.pyplot as plt

def read_results(fname, replicate):
    with open(fname, 'r') as f:
        lines = f.readlines()
        mu_values, std_values = [], []
        mu_times, std_times = [], []
        tm_iter_mu, vl_iter_mu = [], []
        for line in lines:
            splitted = line.split(",")
            mu_values += [float(splitted[0]) for _ in range(replicate)]
            std_values += [float(splitted[1]) for _ in range(replicate)]
            mu_times += [float(splitted[2]) for _ in range(replicate)]
            std_times += [float(splitted[3]) for _ in range(replicate)]
            tm_iter_mu.append(float(splitted[2]))
            vl_iter_mu.append(float(splitted[0]))
    return np.array(mu_values), np.array(std_values), np.array(mu_times), np.array(std_times), np.array(vl_iter_mu), np.array(tm_iter_mu)

def plot_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
#    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    for (label, elems) in results.items():
        mu_v, std_v, mu_t, std_t, mu_v_iter, mu_tm_iter = elems
        ucb_v, lcb_v = mu_v + std_v, mu_v - std_v
        ax1.plot(range(len(mu_v)), mu_v, '-', label=label.replace('_', '-') )
        ax1.fill_between(range(len(mu_v)), abs(lcb_v), ucb_v, alpha=0.6)
        ax2.plot(range(len(mu_t)), mu_t, '-', label=label.replace('_', '-') )
        ax2.fill_between(range(len(mu_t)), abs(mu_t - std_t), mu_t + std_t, alpha=0.6)
        ax3.plot(mu_tm_iter, mu_v_iter, '-', label=label.replace('_', '-') )
        ax3.fill_between(mu_tm_iter, abs(mu_v - std_v), mu_v + std_v, alpha=0.6)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    # ax1.set_xlabel("Stochastic Function Evaluations", fontsize=18)
    # ax1.set_ylabel("$f(x_k) - f(x^*)$", fontsize=18)
    #ax3.set_xscale('log')
    fig.savefig("./linear_reg_values.png", bbox_inches='tight')

m = 100
l = 10
n = 500

svrg_results = read_results('svrg_results', replicate=(n + 2*m)*(l+1))
saga_results = read_results('saga_results', replicate=l+1)
szd_results = read_results('szd_results', replicate=l+1)

plot_results(dict(szd_svrg=svrg_results, szd_saga=saga_results, szd=szd_results))