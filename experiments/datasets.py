
import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.datasets import load_svmlight_file


def load_libsvm_data(datapath, device = 'cpu', dtype = torch.float32):
    X, y = load_svmlight_file(datapath)
    X = torch.from_numpy(X.toarray()).to(dtype)
    y = torch.Tensor(y, device=device)
    return X, y


class SyntheticDataset(Dataset):
    
    def __init__(self, 
                 x_star : torch.Tensor, 
                 n : int, 
                 seed : int = 12131415) -> None:
        self.d = x_star.shape[0]
        self.n = n
        self.dtype = x_star.dtype
        self.device = x_star.device
        self.generator = torch.Generator(device=x_star.device)
        self.generator.manual_seed(seed)
        self.A = torch.randn((n, self.d), dtype = x_star.dtype, device=x_star.device, generator=self.generator)
        self.x_star = x_star
        self.y = self.A @ self.x_star


    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.A[index, :], self.y[index]       
                
    
class LibSVMDataset(Dataset):
    
    def __init__(self, data_path, dtype = torch.float32, device = 'cpu') -> None:        
        super().__init__()
        self.data_path = data_path
        self.X, self.y = load_libsvm_data(data_path, device=device, dtype=dtype)
        self.n = self.X.shape[0]
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.X[index,:], self.y[index]
                
    
    
class MNISTDataset(Dataset):
    
    def __init__(self, dtype = torch.float32, device = 'cpu') -> None:
        super().__init__()
        self.data = torch.utils.data.DataLoader(torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                        ])), batch_size=1)
        self.dtype = dtype
        self.device = device
        self._initialize()
        self.n = self.X.shape[0]

    def _initialize(self):
        self.X, self.y = [], []
        for (img, label) in self.data:
            if len(self.X) > 1000:
                break
            self.X.append(img[0, :, :, :].tolist())
            self.y.append(label.item())
        self.X = torch.Tensor(self.X, device=self.device).to(dtype=self.dtype)            
        self.y = torch.Tensor(self.y)

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.X[index, :, :, :], self.y[index]
        
        
        