from numpy import ndarray, eye
from numpy.linalg import norm, qr
from numpy.random import RandomState

class DirectionGenerator:
    
    def __init__(self, d : int, l : int, seed : int) -> None:
        self.d = d
        self.l = l
        self.rnd_state = RandomState(seed = seed)

    @property
    def shape(self):
        return (self.d, self.l)
        
    def __call__(self) -> ndarray:
        raise NotImplementedError("Call method is implemented in subclasses!")

class SphericalDirections(DirectionGenerator):
    
    def __call__(self) -> ndarray:
        P = self.rnd_state.randn(self.d, self.l)
        P /= norm(P, ord=2, axis=0)
        return P
    
class QRDirections(DirectionGenerator):
    
    def __call__(self) -> ndarray:
        P = self.rnd_state.randn(self.d, self.l)
        return qr(P, mode='reduced')[0]

class CoordinateDirections(DirectionGenerator):
    def __init__(self, d: int, l: int, seed: int) -> None:
        super().__init__(d, l, seed)    
        self.I = eye(d)
        
    def __call__(self) -> ndarray:
        if self.d == self.l:
            return self.I
        indices = self.rnd_state.choice(self.d, self.l, replace=False)
        return self.I[:, indices]
        