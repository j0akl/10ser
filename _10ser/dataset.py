import random
import numpy as np
from math import floor, ceil
from _10ser.tensor import Tensor

class Dataset:
    def __init__(self, x, y):
        # x and y are tensors
        self.data = [(a, b) for a, b in zip(x.data, y.data)]

    def create(self, shuffle=True, split=None):
        # split should be passed as a ratio
        # train/total
        if shuffle == True:
            self.data = random.shuffle(self.data)
        if split is not None:
            section = floor(split*len(self.data))
            self.train = self.data[:section]
            self.val = self.data[len(self.data)-section:]
            # both of these returns should probably be changed to a custom
            # datatype that can be iterated over for training
            return Tensor(self.train), Tensor(self.val)
        return Tensor(self.data)








