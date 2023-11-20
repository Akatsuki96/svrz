
from numpy import ndarray, zeros
from numpy.random import RandomState
from typing import Dict, Optional, Callable
from gzvr.directions import DirectionGenerator


class Optimizer:
    
    def __init__(self, 
                 f : Callable[[ndarray, int], float],
                 n : int,
                 P : DirectionGenerator, 
                 P_full : Optional[DirectionGenerator] = None,
                 seed : int = 121314) -> None:
        self.f = f
        self.n = n
        self.P = P
        self.P_full = P_full if P_full is not None else P
        self.rnd_state = RandomState(seed=seed)
        
    def _approximate_g(self, x : ndarray, P : ndarray, h : float, z : Optional[int] = None):
        fx = self.f(x, z)
        g = zeros(self.P.d)
        for i in range(P.shape[1]):
            g += ((self.f(x + h * P[:, i], z) - fx) / h) * P[:, i]
        return (self.P.shape[0] / self.P.shape[1]) * g, fx
            
    def optimize(self, 
                 x0 : ndarray, 
                 T : int, 
                 gamma : Callable[[int, Optional[ndarray], Optional[float]], float], 
                 h : Callable[[int], float],
                 verbose : bool = False) -> Dict:
        raise NotImplementedError("Optimize method is implemented in subclasses!")