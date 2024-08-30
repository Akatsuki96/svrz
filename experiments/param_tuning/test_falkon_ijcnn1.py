
import torch
import sys

sys.path.append("../")

from datasets import LibSVMDataset


dtype = torch.float32
device = 'cpu'

training_data = LibSVMDataset("/data/mrando/ijcnn1/ijcnn1.tr", dtype=dtype, device=device)

import falkon


options = falkon.FalkonOptions(keops_active="no")



kernel = falkon.kernels.GaussianKernel(sigma=torch.ones((training_data.X.shape[1], ), dtype=dtype, device=device), opt=options)
flk = falkon.Falkon(kernel=kernel, penalty=1e-5, M=500, options=options)


flk.fit(training_data.X, training_data.y)




train_pred = flk.predict(training_data.X).reshape(-1, )

def rmse(true, pred):
    return torch.sqrt(torch.mean((true.reshape(-1, 1) - pred.reshape(-1, 1))**2))

print("Training RMSE: %.3f" % (rmse(train_pred, training_data.y)))

