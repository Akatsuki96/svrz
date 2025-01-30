import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



result_path = sys.argv[1]

algorithms= ["rspgf", "zo_psvrg_gaus", "zo_psvrg_sph", "zo_psvrg_coord", "zo_pspider", "zo_pspider_coord", "opsvrz"] #["opsvrz", "zo_psvrg_coord", "zo_psvrg_gaus", "zo_psvrg_sph", "zo_pspider_coord", "zo_pspider", "rspgf"]

algo_labels = {
    'rspgf' : 'RSPGF',
    'zo_psvrg_gaus' : 'ZO-PSVRG [GausSGE]',
    'zo_psvrg_sph' : 'ZO-PSVRG+ [RandSGE]',
    'zo_psvrg_coord' : 'ZO-PSVRG+ [CoordSGE]',
    'zo_pspider' : 'ZO-PSpider+ [RandSGE]',
    'zo_pspider_coord' : 'ZO-PSpider+ [CoordSGE]',
    'opsvrz' : 'O-PSVRZ',
}


gammas = [0.001, 0.01, 0.1, 1.0]
l_values=[1, 10, 25, 50]
m_values=[50, 100, 150]
T = 100000

def read_results():
    results = {}
    for m in m_values:
        results[m] = {}
        for algo in algorithms:
            results[m][algo] = [[], []] # mean and std of values
            best_result =  None
            ls = l_values if algo != 'rspgf' else [1]
            for l in ls:
                with open(f"{result_path}/param_tuning/{algo}_{l}_{m}_1.log", 'r') as f:
                    last_values = [ float(line.split(',')[1]) + float(line.split(',')[2]) for line in f.readlines() if float(line.split(',')[-1]) in gammas]
                best_idx = np.argmin(last_values)
                if best_result is None or best_result[1] > last_values[best_idx]:
                    best_result = (l, last_values[best_idx], best_idx)
            l, gamma = best_result[0], gammas[best_result[2]]
            with open(f"{result_path}/full_results/{algo}_{l}_{m}_1_{gamma}.log", 'r') as f:
                for line in f.readlines():
                    splitted = line.split(",")
                    mu, std = float(splitted[0]), float(splitted[1])
                    cost_per_iter = int(splitted[-1])
                    results[m][algo][0] += [mu for _ in range(cost_per_iter)]
                    results[m][algo][1] += [std for _ in range(cost_per_iter)]
            results[m][algo][0] = np.array(results[m][algo][0])[:T]
            results[m][algo][1] = np.array(results[m][algo][1])[:T]
    return results

results = read_results()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i in range(len(m_values)):
    result_m = results[m_values[i]]
    axes[i].set_title("$m = {}$".format(m_values[i]), fontsize=20)
    for algo in algorithms:
        full_values, full_stds = result_m[algo]
        axes[i].plot(range(len(full_values)), full_values,'-', label=f"{algo_labels[algo]}", lw=3, rasterized=True)
        axes[i].fill_between(range(len(full_values)), full_values - full_stds, full_values + full_stds, alpha=0.4, rasterized=True)

axes[0].set_ylabel("$F(x^\\tau_0) - \\min \\, F$", fontsize=16)
for i in range(len(axes)):
    axes[i].set_yscale("log")
    axes[i].set_xlabel("# evaluations", fontsize=16)
    axes[i].tick_params(labelsize=12)


axes[0].legend(loc='lower left', fontsize=12)

fig.tight_layout()

fig.savefig("./least_squares.pdf", bbox_inches='tight')
plt.close(fig)