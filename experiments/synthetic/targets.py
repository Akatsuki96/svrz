from typing import Any
import numpy as np


class Target:
    
    def __init__(self, name, d, n, seed :int = 12131415) -> None:
        self.name = name
        self.d = d
        self.n = n
        self.rnd_state = np.random.RandomState(seed=seed)
        
    def __call__(self, x, z = None) -> Any:
        pass
    
class LinearRegression(Target):
    
    def __init__(self, d, n,  seed = 12131415) -> None:
        super().__init__("LinearRegression", d, n, seed=seed)
        self.A = self.rnd_state.randn(n, d)
        self.x_star = np.full((d), 12.0)
        self.y = self.A.dot(self.x_star)
        
    def __call__(self, x, z=None) -> Any:
        if z is None:
            return np.mean(np.square(self.A.dot(x) - self.y), axis=0)
        return np.mean(np.square(self.A[z, :].dot(x) - self.y[z]), axis=0)

class LogisticRegression(Target):
    
    def __init__(self, d, n, seed: int = 12131415) -> None:
        super().__init__("LogisticRegression", d, n, seed)
        C1 = 2*self.rnd_state.randn(n//2, d) 
        C2 = 10 + self.rnd_state.randn(n//2, d)
        self.X = np.vstack((C1, C2))
        self.X = (self.X - self.X.mean()) / self.X.std()
        self.y = np.vstack((np.zeros(n//2) - 1, np.ones(n//2))).reshape(-1)
                
    def __call__(self, w, z=None) -> Any:
        if z is None:
            return np.mean(np.log(1 + np.exp(-self.y * self.X.dot(w))), axis=0)
        return np.mean(np.log(1 + np.exp(-self.y[z] * self.X[z,:].dot(w))), axis=0)
        
        