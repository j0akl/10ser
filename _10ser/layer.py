import numpy as np
from _10ser.tensor import Tensor

# this is a framework for a layer, specific types will inherit
# assumes fully connected layer

class Layer:
    def __init__(self, activation_fn=None):
        self.activation_fn = activation_fn
    def __call__(self, x):
        return self.forward(x)

class Linear(Layer):
    # nodes is a list of nodes
    # possibly refactor prev_layer to inputs
    def __init__(self, n_inputs, width, activation_fn=None):
        super().__init__(activation_fn=activation_fn)
        self.weights = Tensor.rand(width, n_inputs)
        self.biases  = Tensor.rand(width)

    def forward(self, x):
        # if x.data.shape == ():
        #     x = Tensor([x.data])
        outputs = self.weights.matmul(x).data + self.biases.data
        self.outputs = Tensor(outputs)
        if self.activation_fn is not None:
            self.outputs = self.activation_fn(self.outputs)
        return self.outputs

class ReLU(Layer):
    def forward(self, x):
        assert type(x) is Tensor, "input must be tensor"
        output = []
        for i in x.data:
            output.append(np.maximum(0, i))
        self.output = Tensor(output)
        return self.output

    def derivative(self, x):
        if x >= 0:
            return 1
        if x < 0:
            return 0


class Sigmoid(Layer):
    def forward(self, x):
        return 1/(1+np.exp(-x))

