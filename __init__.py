import numpy as np
from _10ser.layer import Linear, ReLU
from _10ser.tensor import Tensor
from _10ser.dataset import Dataset
from _10ser.net import Network
from _10ser.loss import MSELoss



if __name__ == "__main__":
    test = Tensor([1])
    x = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    net = Network([Linear(1, 1, activation_fn=ReLU()), Linear(1, 1, activation_fn=ReLU())], loss=MSELoss())
    print("attempt 1: ", net(Tensor([1])))
    ds = Dataset(x, y)
    x_ds, y_ds = ds.create(shuffle=False, split=0.8)
    net.train(x_ds, 500, 1, .01, y_ds)
    print("attempt 1: ", net(Tensor([1])))
    print("attempt 2: ", net(Tensor([2])))
    print("attempt 3: ", net(Tensor([3])))
    print("attempt 4: ", net(Tensor([4])))

