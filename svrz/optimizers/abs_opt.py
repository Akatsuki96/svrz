
from torch import Tensor, Generator, dtype, float32
from typing import Dict, Optional, Callable
from svrz.directions import DirectionGenerator
from svrz.utils import TargetFunction

class AbsOptimizer:
    
    def __init__(self, 
                 P : DirectionGenerator, 
                 P_full : Optional[DirectionGenerator] = None, 
                 seed : int = 121314, 
                 device : str = 'cpu',
                 dtype : dtype = float32) -> None:
        self.P = P
        self.P_full = P_full if P_full is not None else P
        self.dtype = dtype
        self.device = device
        self.generator = Generator(device=device)
        self.generator.manual_seed(seed)
        
            
    def optimize(self, 
                 f : TargetFunction, # objective function
                 x0 : Tensor,  # initial guess
                 T : int,  # number of iterations
                 gamma : Callable[[int], float],  # stepsize
                 h : Callable[[int], float], # discretization parameter
               ) -> Dict:
        raise NotImplementedError("Optimize method is implemented in subclasses!")