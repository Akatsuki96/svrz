import torch


class DirectionMatrix:
    
    def __init__(self, d, l, seed = 121314, device = 'cpu', dtype=torch.float32) -> None:
        self.d = d
        self.l = l
        self.dtype = dtype
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)      
        
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError()
        
class SphericalDirections(DirectionMatrix):
    
    def __call__(self) -> torch.Tensor:
        A = torch.randn(self.d, self.l, device=self.device, dtype=self.dtype, generator=self.generator)
        A.div_(torch.linalg.norm(A, ord=2, dim=0))
        return A
    
class CoordinateDirections(DirectionMatrix):
        
    def __call__(self) -> torch.Tensor:
        if self.d == self.l:
            return torch.eye(self.d, dtype=self.dtype, device=self.device)
        I = torch.zeros(self.d, self.l, dtype=self.dtype, device=self.device)
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]
        I[inds, range(self.l)] = 1.0
        return I
    
class QRDirections(DirectionMatrix):
    
    def __call__(self) -> torch.Tensor:
        A = torch.randn(self.d, self.l, device=self.device, dtype=self.dtype, generator=self.generator)
        return torch.linalg.qr(A, mode='reduced')[0]
    
    