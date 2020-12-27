import numpy as np

# this is a framework for a layer, specific types will inherit
# assumes fully connected layer
class Layer:
    # nodes is a list of nodes
    # possibly refactor prev_layer to inputs
    def __init__(self, n_inputs, width):
        self.weights = np.random.rand(n_inputs, width)
        self.biases  = np.random.rand(width)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class ReLU:
    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output
