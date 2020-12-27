import numpy as np
from _10ser.tensor import Tensor

class LossFN:
    def __call__(self, x, y):
        return self.calculate(x, y)

class MSELoss(LossFN):
    def calculate(self, x, y):
        # takes a list of tensors, data types need to be cleaned
        # throughout the program
        total = 0
        for i in range(len(x)):
            total = total + (y[i].data-x[i].data)**2
        return total / len(x)

