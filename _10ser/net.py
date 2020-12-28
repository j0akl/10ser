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
                    print("Epoch {}: [{} / {}]".format(epoch+1,
                                                        self.evaluate(val_data),
                                                                len(batches)))
                                                                 # add loss
                                                                 # later
                                                                 # loss_fn()))
            print("Epoch {} Done, Correct: [{} / {}]".format(epoch+1,
                                                             self.evaluate(val_data),
                                                             len(val_data.data)))
    def train_batch(self, batch, lr):
        b0 = [np.zeros(l.biases.data.shape) for l in self.layers]
        w0 = [np.zeros(l.weights.data.shape) for l in self.layers]
        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            b0 = [nb+dnb for nb, dnb in zip(b0, delta_b)]
            w0 = [nw+dnw for nw, dnw in zip(w0, delta_w)]
        for i in range(len(self.layers)):
            # update for line length later
            self.layers[i].weights = Tensor([w-(lr/len(batch))*dw for w, dw in
                             zip(self.layers[i].weights.data, w0[i])])
            self.layers[i].biases = Tensor([b-(lr/len(batch))*db for b, db in
                            zip(self.layers[i].biases.data, b0[i])])

    def backprop(self, x, y):
        # returns a tuple of (db, dw), each is a list for each layer
        b0 = [np.zeros(l.biases.data.shape) for l in self.layers]
        w0 = [np.zeros(l.weights.data.shape) for l in self.layers]
        y = Tensor([y])
        activation = Tensor([x])
        activations = [Tensor([x])]
        zs = []
        for layer in self.layers:
            z = np.dot(layer.weights.data, activation.data) + layer.biases.data
            zs.append(z)
            activation = layer.activation_fn(Tensor(z))
            activations.append(activation)


        f = lambda x: self.layers[-1].activation_fn.derivative(x)
        delta = self.cost_gradient(activations[-1], y) * f(zs[-1])
        b0[-1] = delta
        w0[-1] = np.dot(activations[-2].data.transpose(), delta)

        for i in range(1, len(self.layers)):
            z = zs[-i]
            df = self.layers[-i].activation_fn.derivative(z)
            delta = np.dot(self.layers[-i+1].weights.data.transpose(), delta) * df
            b0[-i] = delta
            w0[-i] = np.dot(delta, activations[-i-1].data.transpose())
        return (b0, w0)

    def evaluate(self, val_data):
        results = [(np.argmax(self.forward(Tensor(x))), y) for x, y in val_data.data]
        return sum(int(x == y) for x, y in results)

    def cost_gradient(self, x, y):
        return (x.data-y.data)





