import numpy as np
from _10ser.layer import Linear, ReLU
from _10ser.loss import MSELoss
from _10ser.tensor import Tensor

if __name__ == "__main__":
    x = [Tensor([i]) for i in range(10)]
    y = [Tensor([i]) for i in range(10)]
    net = Linear(1, 1)
    results = [net(i) for i in x]
    loss_fn = MSELoss()
    loss = loss_fn(results, y)
    print(loss)


