import random
import numpy as np

from network.network import Network

if __name__ == "__main__":
    net = Network([10, 5, 6, 1])
    net([1, 2, 3, 4, 5, 6, 7,  9, 0])
    net.layers_()
