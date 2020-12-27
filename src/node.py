import numpy as np
import random

class Node:

    # inputs is an array of values
    def __init__(self, prev_layer):
        self.inputs = prev_layer
        # size will have to be changed w multidimensional inputs
        self.weights = np.random.rand(prev_layer.size)
        self.bias = random.random()

    def update_inputs(self, inputs):
        self.inputs = inputs

    def get_value(self):
        values = np.multiply(self.inputs, self.weights)
        return sum(values) + self.bias

    def weights(self):
        return self.weights

    def bias(self):
        return self.bias

    def update_weights(self, new_weights):
        self.weights = new_weights

    def update_bias(self, new_bias):
        self.bias = new_bias
