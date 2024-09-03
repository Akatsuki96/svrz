import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import  SGD, AdamW
import time

import tqdm
import torch.nn.functional as F

from torch.utils.data import SubsetRandomSampler
from math import floor

sys.path.append("../")
from net import MNISTNet

out_dir = "./models/mnist"
checkpoint_dir = f"{out_dir}/checkpoints"

os.makedirs(checkpoint_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

seed = 12131415

# Download and load the training data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#validset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)


# Split in training and validation sets

train_size = len(trainset)
valid_size = 0.2
split_point = int(floor(valid_size * train_size))
indices = list(range(train_size))

tr_idx, vl_idx = indices[split_point:], indices[:split_point]

tr_sampler = SubsetRandomSampler(tr_idx, generator = torch.Generator().manual_seed(seed))
vl_sampler = SubsetRandomSampler(vl_idx, generator = torch.Generator().manual_seed(seed))



def write_training_log(fname, tr_err, vl_err):
    with open(fname, 'w') as f:
        for i in range(len(tr_err)):
            f.write("{},{}\n".format(tr_err[i], vl_err[i]))


def train_model(trainloader, validloader, fname, num_epochs = 50, train_temp = 1, init=None, checkpoint_freq = 10, patience =5,  dtype=torch.float32, device='cpu'):
    model = MNISTNet(dtype=dtype, device=device).to(device)
    if init is not None:
        model.load_state_dict(init)
    criterion = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    iterator = tqdm.tqdm(range(num_epochs))
    training_errors, validation_errors = [], []
    best_model = None
    no_improvement = 0
    for epoch in iterator:
        # Training
        model.train(mode=True)
        running_loss = 0.0
        epoch_time = time.time()
        for (i, data) in enumerate(trainloader):
            x, y = data
            x = x.to(device)
            opt.zero_grad()
            preds = model(x)
            loss = criterion(preds.cpu() / train_temp, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        training_errors.append(running_loss / len(trainloader))
        # Validation
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for (i, data) in enumerate(validloader):
                x, y = data
                x = x.to(device)
                preds = model(x)
                loss = criterion(preds.cpu() / train_temp, y)
                running_vloss += loss.item()
        validation_errors.append(running_vloss / len(validloader))
        if best_model is None or best_model[1] > validation_errors[-1]:
            best_model = (model.state_dict(), validation_errors[-1])
            no_improvement = 0
        else:
            no_improvement +=1
        epoch_time = time.time() - epoch_time
        iterator.set_postfix({'epoch' : f'{epoch}/{num_epochs}', 
                              'training error' : running_loss / len(trainloader),
                              'validation error' : running_vloss / len(validloader),
                              'best' : best_model[1],
                              'epoch time' : f"{round(epoch_time, 4)}s"
                              })
        if epoch % checkpoint_freq == 0 and epoch > 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/{fname}_{epoch}")
        if no_improvement == patience:
            break
    torch.save(best_model[0], f"{out_dir}/{fname}")
    return model, training_errors, validation_errors
                
                
def train_distillation(num_epochs = 50, train_temp = 1, batch_size=128, patience = 5, dtype = torch.float32, device='cpu'):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=tr_sampler)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=vl_sampler)

    if not os.path.exists(f"{out_dir}/model_init"):
        train_model(trainloader, validloader, "model_init", num_epochs=1, train_temp=1, patience=patience, dtype=dtype, device=device)
    init_state = torch.load(f"{out_dir}/model_init")
    
    # training teacher model with given temperature
    teacher, teacher_tr_err, teacher_vl_err = train_model(trainloader, validloader, "teacher", num_epochs=num_epochs, train_temp=train_temp, init=init_state, patience=patience, dtype=dtype, device=device)
    # use teacher to get soft-labels
    teacher.eval()
    teacher = teacher.cpu()
    trainset.targets = F.softmax(teacher(trainset.data.view(-1, 1, 28,28).to(dtype)) / train_temp, 1).max(1)[1]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=tr_sampler)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=vl_sampler)
    # train student using soft-labelled dataset
    student, student_tr_err, student_vl_err = train_model(trainloader, validloader, "student", num_epochs=num_epochs, train_temp=train_temp, init=init_state, patience=patience, dtype=dtype, device=device)
    return teacher_tr_err, teacher_vl_err, student_tr_err, student_vl_err
        
num_epochs = 100
train_temp = 100
batch_size = 32
patience = 5
dtype = torch.float32
device = 'cuda'
        
t_tr, t_vl, s_tr, s_vl = train_distillation(num_epochs=num_epochs, train_temp=train_temp, batch_size=batch_size, patience=patience, dtype=dtype, device=device)

write_training_log(f"{out_dir}/teacher_training.log", t_tr, t_vl)
write_training_log(f"{out_dir}/student_training.log", s_tr, s_vl)
