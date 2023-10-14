import numpy as np


class OptResult:
    
    def __init__(self) -> None:
        self.f_values = []
        
    def add_result(self, fx):
        self.f_values.append(fx)
        