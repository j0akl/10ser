import numpy as np
import _10ser.layer as layer
from _10ser.tensor import Tensor
from _10ser.loss import MSELoss

class Network:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        current_x = x
        for i in self.layers:
            current_x = i(current_x)
        return current_x

    def backprop(self, x, y):
        pass


