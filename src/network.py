import numpy as np

from .layer import Layer


class Network:

    def __init__(self, layers):
        # layers done in sequential order
        # layers should be an array of integers specifing # nodes
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
        print(inputs)
        self.inputs = np.array(inputs)
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].update_inputs(inputs)
            else:
                self.layers[i].update_inputs(self.layers[i-1]())

    def layers_(self):
        for i in range(len(self.layers)):
            print("--- " + str(i)+ " ---")
            print("inputs: ", self.layers[i].inputs)
            print("input size: ", self.layers[i].inputs.size)
            print("outputs: ", self.layers[i].generate_outputs())
            print("output size: ", self.layers[i].generate_outputs().size)


    def nodes_(self):
        for layer in self.layers:
            for node in layer.nodes:
                print("-----")
                print("inputs: ", node.inputs)
                print("val: ", node.get_value())
                print("weights: ", node.weights)
                print("bias: ", node.bias)

