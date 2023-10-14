import numpy as np


class TargetFunction:
    
    def __init__(self, d, x_star, f_star) -> None:
        self.d = d
        self.x_star = x_star
        self.f_star = f_star
        
    def __call__(self, x, z = None) -> float:
        pass
    
    
