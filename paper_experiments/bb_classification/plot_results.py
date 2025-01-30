import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



result_path = sys.argv[1]
dataset_name = sys.argv[2]

d = int(sys.argv[3])

plot_legenda = int(sys.argv[4]) == 1

dname_to_label = {
    'australian' : 'Australian',
    'phishing' : "Phishing",
    'mushrooms' : 'Mushrooms',
    'german.numer' : 'German Numer',
    'splice' : "Splice"
}



algorithms= ["rspgf", "zo_psvrg_gaus", "zo_psvrg_sph", "zo_psvrg_coord", "zo_pspider", "zo_pspider_coord", "opsvrz"] 

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
l_values=[d // i for i in [10, 5, 2, 1]]

m = 50

T = 10000000

def read_results():
    results = {}
    results[m] = {}
    for algo in algorithms:
        results[m][algo] = [[], []] # mean and std of values
        best_result =  None
        ls = l_values 
        if algo == 'rspgf':
            ls = [1]
        elif algo == 'zo_psvrg_coord' or algo == 'zo_pspider_coord':
            ls = [d]
        for l in ls:
            with open(f"{result_path}/{dataset_name}/param_tuning/{algo}_{l}_{m}_1.log", 'r') as f:
                last_values = [ float(line.split(',')[1]) + float(line.split(',')[2]) for line in f.readlines() if float(line.split(',')[-1]) in gammas]
            best_idx = np.argmin(last_values)
            if best_result is None or best_result[1] > last_values[best_idx]:
                best_result = (l, last_values[best_idx], best_idx)
        if algo == 'rspgf':
            best_result = (1, last_values[best_idx], best_idx)
        l, gamma = best_result[0], gammas[best_result[2]]
        print(f"ALGO: {algo}\tl: {l}\tgamma: {gamma}")
        with open(f"{result_path}/{dataset_name}/full_results/{algo}_{l}_{m}_1_{gamma}.log", 'r') as f:
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

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

result_m = results[m]
ax.set_title(dname_to_label[dataset_name], fontsize=20)
for algo in algorithms:
    full_values, full_stds = result_m[algo]
    ax.plot(range(len(full_values)), full_values,'-', label=f"{algo_labels[algo]}", lw=3, rasterized=True)
    ax.fill_between(range(len(full_values)), full_values - full_stds, full_values + full_stds, alpha=0.4, rasterized=True)

ax.set_ylabel("$F(x^\\tau_0) - \\min \\, F$", fontsize=16)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("# evaluations", fontsize=16)
ax.tick_params(labelsize=12)

if plot_legenda:
    ax.legend(loc='upper right', fontsize=12)

fig.tight_layout()

fig.savefig(f"./bb_class_{dataset_name}.pdf", bbox_inches='tight')
plt.close(fig)