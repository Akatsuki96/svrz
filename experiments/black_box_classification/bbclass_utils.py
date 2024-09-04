import torch
import numpy as np 


def test_optimizer(target, name, optimizer, x0, T, m, gamma, h, test_set, cost_per_iter = None, reps = 10, out_path='./'):
    values, times = [], []
    te_error = []
    for _ in range(reps):
        if m is None:
            result = optimizer.optimize(target, x0 = x0, T = T, gamma = gamma, h = h)
        else:
            result = optimizer.optimize(target, x0 = x0, T = T, m = m, gamma = gamma, h = h)
        values.append(result['f_values'])
        times.append(result['it_times'])
        w = result['x']
        err = ((torch.sigmoid(torch.matmul(w, test_set.X.T)) > 0.5).to(torch.int64) != test_set.y).sum() / test_set.y.shape[0]
        te_error.append(err.item())

    print("-"*44)
    values = np.array(values).reshape(reps, -1)
    times = np.array(times).reshape(reps, -1).cumsum(1)
    ris = {
        'values' : (values.mean(0), values.std(0)),
        'times' : (times.mean(0), times.std(0))
    }
    if cost_per_iter is None:
        cost_per_iter = result['l_values']
            
    with open(f"{out_path}/{name}.log", 'w') as f:
        for i in range(len(ris['values'][0])):
            cost = cost_per_iter[i] if isinstance(cost_per_iter, list) else cost_per_iter
            f.write(f"{ris['values'][0][i]},{ris['values'][1][i]},{ris['times'][0][i]},{ris['times'][1][i]},{cost}\n")                
    with open(f"{out_path}/{name}_test_error.log", 'w') as f:
        f.write("{},{}\n".format(np.mean(te_error), np.std(te_error)))
    print(f"[{name}] Test error: {np.mean(te_error)} +/- {np.std(te_error)}")
    return None 
