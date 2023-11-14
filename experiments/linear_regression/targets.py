import torch
import numpy as np

class TargetFunction:
    
    def __init__(self, d, x_star, f_star) -> None:
        self.d = d
        self.x_star = x_star
        self.f_star = f_star
        
    def __call__(self, x, z = None) -> float:
        pass


class LinearRegression(TargetFunction):
    
    def __init__(self, d, n, seed = 121314) -> None:
        self.d = d
        self.n = n
        self.rnd_state = np.random.RandomState(seed = seed)
        self.X = 10 * self.rnd_state.rand(n, d)
        self.w_star = self.rnd_state.rand(self.d)
        self.y = self.X @ self.w_star
        super().__init__(d, self.w_star, 0.0)
        
    def __call__(self, w, z=None) -> float:
        if z is None:
            return  (1/self.n) * np.sum(np.square(self.X @ w - self.y)) 
        return np.square(self.X[z, :] @ w - self.y[z])
    