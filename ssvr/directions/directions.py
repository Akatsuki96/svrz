from typing import Any
import numpy as np


class DirectionMatrix:
    
    def __init__(self, d, l, seed = 121314) -> None:
        self.d = d
        self.l = l
        self.rnd_state = np.random.RandomState(seed = seed)
        
        
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()
        
class SphericalDirections(DirectionMatrix):
    
    def __call__(self) -> np.ndarray:
        A = self.rnd_state.randn(self.d, self.l)
        return A / np.linalg.norm(A, ord=2, axis=0)
    
class QRDirections(DirectionMatrix):
    
    def __call__(self) -> np.ndarray:
        A = self.rnd_state.randn(self.d, self.l)
        return np.linalg.qr(A, mode='reduced')[0]