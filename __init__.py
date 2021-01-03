import numpy as np
from _10ser.layer import Linear, ReLU
from _10ser.dataset import Dataset
from _10ser.net import Network
from _10ser.loss import MSELoss



if __name__ == "__main__":
    test = np.array([1])
    x = np.array([i for i in range(100)])
    y = np.array([i*2 for i in range(100)])

    net = Network([Linear(1, 1, activation_fn=ReLU())], loss=MSELoss())
    print("attempt 1: ", net(np.array([1])))
    ds = Dataset(x, y)
    x_ds, y_ds = ds.create(shuffle=False, split=0.8)
    print(x_ds)
    net.train(x_ds, 1, 2, .1, y_ds)
    for i in range(10):
        print(y[i])
        print("out: ", net(x[i]))
