import random
import numpy as np

"""
TODO:
    - need to add support for multiple dimesions
    - optimize (after its working)
    - 
"""

class Node:

    # inputs is an array of values
    def __init__(self, input_nodes):
        self.inputs = input_nodes
        self.weights = np.random.rand(len(self.inputs))
        self.bias = random.random()

    def update_inputs(self, inputs):
        self.inputs = inputs

    def get_value(self):
        value = 0
        for i in range(len(self.inputs)):
            value += self.inputs[i]*self.weights[i]
        return value + self.bias

    def weights(self):
        return self.weights

    def bias(self):
        return self.bias

    def update_weights(self, new_weights):
        self.weights = new_weights

    def update_bias(self, new_bias):
        self.bias = [new_bias]

# this is a framework for a layer, specific types will inherit
# assumes fully connected layer
class Layer:

    # nodes is a list of nodes
    def __init__(self, length):
        self.nodes = [Node([]))]

    def pass_inputs_to_nodes(self, prev_layer):
        # this can be updated for speed later, just focus on concept atm
        # prev_layer is the gen_outputs of the prev layer
        for i in range(len(nodes)):
                nodes[i].update_inputs(prev_layer)

    def generate_outputs(self):
        values = []
        for i in range(len(nodes)):
            values.append(self.nodes[i].get_value())

class Network:

    def __init__(self, layers):
        # layers done in sequential order
        # layers should be an array of integers
        # length of layers is num layers, value at i
        # is the number of neurons in that layer
        self.layers = [Layer(num_nodes) for num_nodes in layers]










if __name__ == "__main__":
    node = Node([1, 2, 3])
