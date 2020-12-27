import random
import numpy as np

"""
TODO:
    - need to add support for multiple dimesions
    - optimize (after its working)
    - break into individual files
"""

class Node:

    # inputs is an array of values
    def __init__(self, prev_layer):
        self.inputs = np.zeros_like(len(prev_layer))
        # size will have to be changed w multidimensional inputs
        self.weights = np.random.rand(len(prev_layer))
        self.bias = random.random()

    def update_inputs(self, inputs):
        self.inputs = inputs

    def get_value(self):
        value = 0
        for i in range(self.inputs.size):
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
    def __init__(self, length, prev_layer):
        self.prev = prev_layer
        self.nodes = [Node(self.prev) for i in range(length)]

    def update_prev(self, new_prev):
        self.prev = new_prev

    def pass_inputs_to_nodes(self):
        # this can be updated for speed later, just focus on concept atm
        # prev_layer is the gen_outputs of the prev layer
        for i in range(len(nodes)):
                nodes[i].update_inputs(self.prev)

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
        self.input_layer = np.zeros_like(layers[0])
        self.layers = []
        for i in range(len(layers)):
            if i != 0:
                self.layers.append(Layer(layers[i],
                                         np.zeros_like(range(layers[i-1]))))




if __name__ == "__main__":
    net = Network([2, 3, 4, 1])
    for i in net.layers:
        for j in i.nodes:
            print(j.weights)

