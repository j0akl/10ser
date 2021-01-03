import numpy as np
# this is a framework for a layer, specific types will inherit
# assumes fully connected layer

class Layer:
    def __init__(self, activation_fn=None):
        self.activation_fn = activation_fn
    def __call__(self, x):
        return self.forward(x)

class Input(Layer):
    def __init__(self, n_inputs):
        self.inputs = np.zeros_like(range(n_inputs))

    def forward(self, x):
        self.inputs = x
        return self.inputs


class Linear(Layer):
    # nodes is a list of nodes
    # possibly refactor prev_layer to inputs
    def __init__(self, n_inputs, width, activation_fn=None):
        super().__init__(activation_fn=activation_fn)
        self.weights = np.ones((width, n_inputs))
        self.biases  = np.zeros(width)

    def forward(self, x):
        self.outputs = np.dot(self.weights, x) + self.biases
        if self.activation_fn is not None:
            self.outputs = self.activation_fn(self.outputs)
        return self.outputs

class ReLU(Layer):
    def forward(self, x):
        output = []
        if len(x) == 1:
            output.append(np.maximum(0, 1))
        else:
            for i in x:
                output.append(np.maximum(0, i))
        self.output = output
        return self.output

    def derivative(self, x):
        if x >= 0:
            return 1
        if x < 0:
            return 0


class Sigmoid(Layer):
    def forward(self, x):
        return 1/(1+np.exp(-x))

