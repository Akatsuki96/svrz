import torch
import torch.nn as nn
import torch.nn.functional as F



class MNISTNet(nn.Module):
    def __init__(self, dtype : torch.dtype = torch.float32, device : str = 'cpu'):
        super(MNISTNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dtype=dtype, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),            
            nn.Linear(1024, 200, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(200, 200, dtype=dtype, device=device),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(200, 10, dtype=dtype, device=device),
        )


    def forward(self, x):
        return self.net(x)
