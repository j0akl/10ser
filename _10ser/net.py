import numpy as np
import _10ser.layer as layer
from _10ser.tensor import Tensor
from _10ser.loss import MSELoss

class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss_fn = loss

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        current_x = x
        for i in self.layers:
            current_x = i(current_x)
        return current_x

    def train(self, x, epochs, batch_size, lr, val_data=None):
        # train data (x) must be provided as a tuple tensor from Dataset
        # if val is none, will skip validation after epoch
        batches = [x.data[k:k+batch_size] for k in range(len(x.data))]
        for epoch in range(epochs):
            for batch in batches:
                self.train_batch(batch, lr)
                if val_data is not None:
                    print("Epoch {}: [{} / {}] Loss: {} ".format(epoch+1,
                                                                 batches.index(batch+1),
                                                                 len(batches)))
                                                                 # add loss
                                                                 # later
                                                                 # loss_fn()))
            print("Epoch {} Done, Correct: [{} / {}]".format(epoch+1,
                                                             self.accuracy(val_data),
                                                             len(val_data)))
    def train_batch(self, batch, lr):
        for layer in reversed(self.layers):
            b = layer.biases
            w = layer.weights
            b_0 = np.zeros_like(b)
            w_0 = np.zeros_like(w)





