import numpy as np
from .node import Node

# this is a framework for a layer, specific types will inherit
# assumes fully connected layer
class Layer:

    # nodes is a list of nodes
    # possibly refactor prev_layer to inputs
    def __init__(self, length, prev_layer):
        self.inputs = prev_layer
        self.nodes = [Node(self.inputs) for i in range(length)]

    def __call__(self):
        return self.generate_outputs()


    def update_inputs(self, new_inputs):
        self.inputes = new_inputs
        self.pass_inputs_to_nodes()

    def update_node_weights(self, weights):
        # weights should be an array the same shape
        # as the weights of the given layer, this wont
        # really be useful later on (?)
        for i in range(len(self.nodes)):
            self.nodes[i].update_weights(weights[i])

    def pass_inputs_to_nodes(self):
        # this can be updated for speed later, just focus on concept atm
        # prev_layer is the gen_outputs of the prev layer
        for i in range(len(self.nodes)):
            self.nodes[i].update_inputs(self.inputs)


    def generate_outputs(self):
        values = []
        for node in self.nodes:
            values.append(node.get_value())
        return np.array(values)
