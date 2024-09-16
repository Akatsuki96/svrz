import numpy as np

import matplotlib.pyplot as plt


directory = "./results/changing_l"

d = 50
m = 50
num_directions = [1] + [i for i in range(5, d + 5, 5)]
gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
inner_iters = [5, 25, 50]
opt_names = ['osvrz']


# methods = [
#     ('O-SVRZ $[\\ell = 1]$' ,  f'{directory}/osvrz_1.log'),
#     ('O-SVRZ $[\\ell = 5]$' ,  f'{directory}/osvrz_5.log'),
#     ('O-SVRZ $[\\ell = 10]$' , f'{directory}/osvrz_10.log'),
#     ('O-SVRZ $[\\ell = 15]$' , f'{directory}/osvrz_15.log'),
#     ('O-SVRZ $[\\ell = 20]$' , f'{directory}/osvrz_20.log'),
#     ('O-SVRZ $[\\ell = 25]$' , f'{directory}/osvrz_25.log'),
#     ('O-SVRZ $[\\ell = 30]$' , f'{directory}/osvrz_30.log'),
#     ('O-SVRZ $[\\ell = 35]$' , f'{directory}/osvrz_35.log'),
#     ('O-SVRZ $[\\ell = 40]$' , f'{directory}/osvrz_40.log'),
#     ('O-SVRZ $[\\ell = 45]$' , f'{directory}/osvrz_45.log'),
#     ('O-SVRZ $[\\ell = 50]$' , f'{directory}/osvrz_50.log'),
# ]

def read_result(budget, path):
    mu_val, std_val = [], []
    mu_tim, std_tim = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            splitted = line.split(",")
            mu_val += [float(splitted[0]) for _ in range(int(splitted[-1]))]
            std_val += [float(splitted[1]) for _ in range(int(splitted[-1]))]
            mu_tim += [float(splitted[2]) for _ in range(int(splitted[-1]))]
            std_tim += [float(splitted[3]) for _ in range(int(splitted[-1]))]
            if len(mu_val) > budget:
                break
    mu_val, std_val = np.array(mu_val)[:budget], np.array(std_val)[:budget]
    mu_tim, std_tim = np.array(mu_tim)[:budget], np.array(std_tim)[:budget]
    return dict(values=(mu_val, std_val), times=(mu_tim, std_tim))


gamma = 0.01
m = 100

for m in [75]:

    methods = [(f'O-SVRZ $[\\ell = {l}]$' ,  f'{directory}/osvrz_{l}_{m}.log') for l in num_directions ]



    budget = 250000



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle(f"$m = {m}$")

    ax1.set_title("Function values")
    ax2.set_title("Cumulative Computational Cost")

    for (label, path) in methods:
        ris = read_result(budget, path)
        mu_values, std_values = ris['values']
        mu_times,  std_times  = ris['times']
        
        ax1.plot(range(len(mu_values)), mu_values, '-', lw=3, label=label)
        ax1.fill_between(range(len(mu_values)), mu_values - std_values, mu_values + std_values, alpha=0.6)
        ax2.plot(range(len(mu_times)), mu_times, '-', lw=3, label=label)
        ax2.fill_between(range(len(mu_times)), mu_times - std_times, mu_times + std_times, alpha=0.6)
        ris = None

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.legend(loc='upper right')

    ax1.set_xlabel("# stochastic function values", fontsize=12)
    ax1.set_ylabel("$F(x^\\tau) - F(x^*)$", fontsize=12)

    ax2.set_xlabel("# stochastic function values", fontsize=12)
    ax2.set_ylabel("Cost (s)", fontsize=12)

    fig.tight_layout()
    fig.savefig(f"changing_l_{gamma}_{m}.png", bbox_inches='tight')
    plt.close(fig)