import random
import numpy as np

from src.network import Network

if __name__ == "__main__":
    net = Network([10, 15, 6, 9])
    net.layers_()
    net([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    net.layers_()
