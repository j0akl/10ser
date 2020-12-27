import random
import numpy as np

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
        value = 0
        for i in range(self.inputs.size - 1):
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

    def pass_inputs_to_nodes(self, inputs):
        # this can be updated for speed later, just focus on concept atm
        # prev_layer is the gen_outputs of the prev layer
        for node in self.nodes:
                node.update_inputs(inputs)

    def generate_outputs(self):
        values = []
        for node in self.nodes:
            values.append(node.get_value())
        return np.array(values)


class Network:

    def __init__(self, layers):
        # layers done in sequential order
        # layers should be an array of integers
        # length of layers is num layers, value at i
        # is the number of neurons in that layer
        self.inputs = np.zeros_like(range(layers[0]), dtype=np.float64)

        # TODO
        # layers can be changed in inherited classes to accept different types
        # maybe include the type in the input when the net is created
        # there is probably a better way to do this, maybe have each network
        # be created by the user like pytorch
        # works for now tho
        self.layers = [Layer(layers[0], self.inputs)]
        for i in range(len(layers)):
            if i != 0:
                self.layers.append(Layer(layers[i],
                                         self.layers[i-1].generate_outputs()))

    def __call__(self, inputs):
        # inputs is an array length same as first layer
        self.inputs = np.array(inputs)
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].pass_inputs_to_nodes(self.inputs)
            else:
                self.layers[i].pass_inputs_to_nodes(self.layers[i-1].generate_outputs())

    def print_net(self):
        for i in self.layers:
            for j in i.nodes:
                print("-----")
                print("inputs: ", j.inputs)
                print("val: ", j.get_value())
                print("weights: ", j.weights)
                print("bias: ", j.bias)


if __name__ == "__main__":
    net = Network([2, 3, 4, 1])

    net.print_net()
    print("________________")


    net([1, 2])

    net.print_net()


