import numpy as np
from _10ser.layer import Linear, ReLU
from _10ser.loss import MSELoss
from _10ser.tensor import Tensor
from _10ser.net import Network

if __name__ == "__main__":
    net = Network([Linear(5, 3, activation_fn=ReLU())])
    inputs = Tensor([1, 2, 3, 4, 5])
    print(net(inputs))




