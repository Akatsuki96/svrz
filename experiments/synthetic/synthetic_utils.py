import numpy as np




def test_optimizer(target, optimizer, x0, T, m, gamma, h, cost_per_iter = None, reps = 10):
    values, times = [], []
    for _ in range(reps):
        if m is None:
            result = optimizer.optimize(target, x0 = x0, T = T, gamma = gamma, h = h)
        else:
            result = optimizer.optimize(target, x0 = x0, T = T, m = m, gamma = gamma, h = h)
        values.append(result['f_values'])
        times.append(result['it_times'])
    print("-"*44)
    values = np.array(values).reshape(reps, -1)
    times = np.array(times).reshape(reps, -1).cumsum(1)
    ris = {
        'values' : (values.mean(0), values.std(0)),
        'times' : (times.mean(0), times.std(0))
    }
    if cost_per_iter is None:
        cost_per_iter = result['l_values']
    ris['cost_per_iter'] = cost_per_iter           
    return ris
