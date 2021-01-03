import numpy as np
import _10ser.layer as layer
from _10ser.loss import MSELoss

class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss_fn = loss
        self.weights = []
        self.biases = []
        for i in self.layers:
            self.weights.append(i.weights)
            self.biases.append(i.biases)
        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        current_x = x
        for w, b in zip(self.weights, self.biases):
            current_x = np.dot(w, current_x) + b
        return current_x

    def train(self, x, epochs, batch_size, lr, val_data=None):
        # train data (x) must be provided as a tuple tensor from Dataset
        # if val is none, will skip validation after epoch
        batches = [x[k:k+batch_size] for k in range(len(x))]
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
                                                             len(val_data)))
    def train_batch(self, batch, lr):
        b0 = [np.zeros_like(b) for b in self.biases]
        w0 = [np.zeros_like(w) for w in self.weights]
        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            b0 = [nb+dnb for nb, dnb in zip(b0, delta_b)]
            w0 = [nw+dnw for nw, dnw in zip(w0, delta_w)]
        for i in range(len(self.weights)):
            # update for line length later
            print("weights: ", self.weights)
            print("biases: ", self.biases)
            self.weights = [w-(lr/len(batch))*dw for w, dw in
                             zip(self.weights[i], w0[i])]
            self.biases = [b-(lr/len(batch))*db for b, db in
                            zip(self.biases[i], b0[i])]

    def backprop(self, x, y):
        # returns a tuple of (db, dw), each is a list for each layer
        b0 = [np.zeros(l.biases.shape) for l in self.layers]
        w0 = [np.zeros(l.weights.shape) for l in self.layers]
        activation = x
        activations = [x]
        zs = []
        for w, b, layer in zip(self.weights, self.biases, self.layers):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = layer.activation_fn(z)
            activations.append(activation)


        f = lambda x: self.layers[-1].activation_fn.derivative(x)
        delta = self.cost_gradient(activations[-1], y) * f(zs[-1])
        b0[-1] = delta
        w0[-1] = np.dot(np.array(activations[-2]).transpose(), delta)

        for i in range(1, len(self.layers)):
            z = zs[-i]
            df = self.layers[-i].activation_fn.derivative(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * df
            b0[-i] = delta
            w0[-i] = np.dot(delta, np.array(activations[-i-1]).transpose())
        return (b0, w0)

    def evaluate(self, val_data):
        results = [(np.argmax(self.forward(x)), y) for x, y in val_data]
        return sum(int(x == y) for x, y in results)

    def cost_gradient(self, x, y):
        return (x-y)





