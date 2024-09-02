import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, input_shape, params, dtype : torch.dtype = torch.float32):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=params[0], kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()


        self.conv3 = nn.Conv2d(in_channels=params[1], out_channels=params[2], kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=params[2], out_channels=params[3], kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_conv_output(input_shape), params[4])
        self.fc2 = nn.Linear(params[4], params[5])
        self.fc3 = nn.Linear(params[5], 10)
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output(self, shape):
        # Create a dummy tensor with the same shape as the input to pass through the conv layers
        dummy_input = torch.zeros(1, *shape)
        output = self.conv4(self.conv3(self.pool1(self.conv2(self.conv1(dummy_input)))))
        output = self.pool2(output)
        return output.numel()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)
        return x