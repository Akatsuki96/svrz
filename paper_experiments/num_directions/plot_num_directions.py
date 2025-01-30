import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

RESULTS_PATH = "/data/mrando/svrz_results/num_directions"
fontsize_ticks = 12

budget = 100000

def read_full_result(path, name, m, l, b, gamma, mu, L):
    mu_values, std_values = [], []
#    mu_times, std_times = [], []
    #        with open(f"{out_path}/full_results/{name}_{m}_{l}_{b}_{gamma}_{mu}_{L}.log", 'a') as f:
    with open(f"{path}/full_results/{name}_{m}_{l}_{b}_{gamma}_{mu}_{L}.log", 'r') as f:
        #                f.write(f"{mu_vals[i]},{std_vals[i]},{mu_time[i]},{std_time[i]},{cost}\n")                
        for line in f.readlines():
            splitted = line.split(",")
            num_evals = int(splitted[-1])
            mu_values += [float(splitted[0]) for _ in range(num_evals)]
            std_values += [float(splitted[1]) for _ in range(num_evals)]
#            mu_times  += [float(splitted[2]) for _ in range(num_evals)]
#            std_times += [float(splitted[3]) for _ in range(num_evals)]

    mu_values = np.array(mu_values)[:budget]
    std_values = np.array(std_values)[:budget]
#    mu_times = np.array(mu_times)[:budget]
#    std_times = np.array(std_times)[:budget]
    
    return mu_values, std_values #, mu_times, std_times

def read_paramtuning_result(path, name, m, l, b):
    mu_values, std_values = [], []
    
    with open(f"{path}/param_tuning/{name}_{m}_{l}_{b}.log", 'r') as f:
        for line in f.readlines():
            splitted = line.split(",")
            f0 = float(splitted[0])
            mu_v = float(splitted[1]) if float(splitted[1]) == float(splitted[1])  and float(splitted[1]) + float(splitted[2]) <= f0  else f0
            std_v = float(splitted[2]) if float(splitted[1]) == float(splitted[1]) and float(splitted[1]) + float(splitted[2]) <= f0  else 0.0
            mu_values.append(mu_v)
            std_values.append(std_v)
    return np.array(mu_values), np.array(std_values)#, np.array(mu_times), np.array(std_times)



gammas = np.logspace(-4, 0, 50)
l_values = [1, 10, 20, 30, 40, 50]
m_values = [50, 100, 150]
name = 'opsvrz'


print("[--] Plotting comparison on l")  

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(len(m_values)):
    axes[0, i].set_title("$m = {}$".format(m_values[i]), fontsize=20)
    for l in l_values:
        print("\tl = {}\tm = {}".format(l, m_values[i]))
        values, stds = read_paramtuning_result(RESULTS_PATH, name, m=m_values[i], l=l, b = 1)
        axes[1][i].plot(gammas, values,'o-', label='$\\ell = {}$'.format(l), rasterized=True)
        axes[1][i].fill_between(gammas, values- stds, values + stds, alpha=0.4, rasterized=True)
        best_gamma =  gammas[np.argmin(values)] 
        full_values, full_stds = read_full_result(RESULTS_PATH, name, m = m_values[i], l = l, b = 1, gamma=best_gamma, mu=1, L = 10)

        best_gamma =  gammas[np.argmin(values)]
        full_values, full_stds = read_full_result(RESULTS_PATH, name, m = m_values[i], l = l, b = 1, gamma=best_gamma, mu=1, L = 10)
        axes[0][i].plot(range(len(full_values)), full_values,'-', label='$\\ell = {}$'.format(l), lw=3, rasterized=True)
        axes[0][i].fill_between(range(len(full_values)), full_values - full_stds, full_values + full_stds, alpha=0.4, rasterized=True)

for i in range(len(axes)):
    for j in range(len(axes[i])):
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel("$\gamma$", fontsize=16)
        axes[0, j].set_xlabel("# evaluations", fontsize=16)
        axes[0, 0].set_ylabel("$F(x^\\tau_0) - \\min \\, F$", fontsize=16)
        axes[1, 0].set_ylabel("$F(x^T_0) - \\min \\, F$", fontsize=16)
        axes[i, j].set_yscale("log")
        axes[i, j].tick_params(labelsize=fontsize_ticks)


axes[0,0].legend(loc='lower left', fontsize=12)

fig.tight_layout()



fig.savefig("./changing_params_l.pdf", bbox_inches='tight')
plt.close(fig)
         
