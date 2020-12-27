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
        b0 = [np.zeros_like(l.biases) for l in self.layers]
        w0 = [np.zeros_like(l.weights) for l in self.layers]
        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            b0 = [nb+dnb for nb, dnb in zip(b0, delta_b)]
            w0 = [nw+dnw for nw, dnw in zip(w0, delta_w)]
        for i in range(len(self.layers)).layers:
            # update for line length later
            layer.weights = [w-(lr/len(batch))*dw for w, dw in zip(self.layers[i].weights, w0[i])]
            layer.biases = [b-(lr/len(batch))*db for b, db in zip(self.layers[i].biases, b0[i])]

    def backprop(self, x, y):
        # returns a tuple of (db, dw), each is a list for each layer
        b0 = [np.zeros_like(l.biases) for l in self.layers]
        w0 = [np.zeros_like(l.weights) for l in self.layers]
        activation = x
        activations = [x]
        zs = []
        for layer in self.layers:
            layer.forward(activation)
            z = layer.outputs
            zs.append(z)
            activation = layer.post_activation
            activations.append(activation)


        delta = self.cost_gradient(activations[-1], y) * layer.activation_fn.derivative(zs[-1])
        b0[-1] = delta
        w0[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, len(self.layers)):
            z = zs[-i]
            df = self.layers[-i].activation_fn.derivative(z)
            delta = np.dot(self.layers[-i+1].weights.transpose(), delta) * df
            b0[-i] = delta
            w0[-i] = np.dot(delta, activations[-i+1].transpose())
        return (b0, w0)

    def evaluate(self, val_data):
        results = [(np.argmax(self.forward(x)), y) for x, y in val_data]
        return sum(int(x == y) for x, y in results)

    def cost_gradient(self, x, y):
        return (x-y)





