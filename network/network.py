import numpy as np
from layer import Layer, ReLU

class Net:
    def __init__(self):
        # networks built by the user like in pytorch
        # this is an example
        self.linear1 = Layer(5, 6)
        self.ReLU1 = ReLU()
        self.linear2 = Layer(6, 1)
        self.ReLU2 = ReLU()

    def __call__(self, x):
        # inputs must be same shape as first layer
        x = self.linear1.forward(x)
        x = self.ReLU1.forward(x)
        x = self.linear2.forward(x)
        output = self.ReLU2.forward(x)
        return output

if __name__ == "__main__":
    net = Net()
    inputs = np.array([1, 2, 8, 4, 5])
    print(net(inputs))
