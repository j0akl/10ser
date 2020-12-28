import numpy as np
import matplotlib.pyplot as plt
from _10ser.dataset import Dataset
from _10ser.tensor import Tensor
from _10ser.layer import ReLU, Linear
from _10ser.loss import MSELoss
from _10ser.net import Network

train_data = np.loadtxt("mnist_data/mnist_train.csv", delimiter=',')
test_data = np.loadtxt("mnist_data/mnist_test.csv", delimiter=',')

x_train = []
y_train = []
for i in range(len(train_data)):
    x_train.append(train_data[i][:-1])
    y_train.append(train_data[i][-1])

x_train = Tensor(x_train)
y_train = Tensor(y_train)

ds = Dataset(x_train, y_train)
train_ds, val_ds = ds.create(split=0.8)

net = Network([Linear(1, 784, activation_fn=ReLU()), 
               Linear(784, 16, activation_fn=ReLU()),
               Linear(16, 16, activation_fn=ReLU()),
               Linear(16, 10, activation_fn=ReLU())], MSELoss())
print(net(Tensor(x_train.data[0])))
# net.train(train_ds, 5, 32, 0.001, val_data=val_ds)
