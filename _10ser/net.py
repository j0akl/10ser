import numpy as np
import _10ser.layer as layer
from _10ser.tensor import Tensor
from _10ser.loss import MSELoss

class Network:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for i in self.layers:
            pass

    def backprop(self, x, y):
        pass


