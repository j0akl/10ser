import numpy as np
from _10ser.tensor import Tensor

# this is a framework for a layer, specific types will inherit
# assumes fully connected layer

class Layer:
    # provides children with __call__ to use
    # self.forward method, more added later
    def __call__(self, x):
        return self.forward(x)

class Linear(Layer):
    # nodes is a list of nodes
    # possibly refactor prev_layer to inputs
    def __init__(self, n_inputs, width):
        self.weights = Tensor.rand(n_inputs, width)
        self.biases  = Tensor.rand(width)
        self.outputs = []

    def forward(self, x):
        self.outputs = self.weights.matmul(x).data + self.biases.data
        return Tensor(self.outputs)

class ReLU(Layer):
    def forward(self, x):
        assert type(x) is Tensor, "input must be tensor"
        output = []
        for i in x.data:
            output.append(np.maximum(0, i))
        self.output = Tensor(output)
        return self.output

class Sigmoid(Layer):
    def forward(self, x):
        return 1/(1+np.exp(-x))

