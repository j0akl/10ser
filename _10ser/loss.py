import numpy as np

class LossFN:
    def __call__(self, x, y):
        return self.calculate(x, y)

class MSELoss(LossFN):
    def calculate(self, outputs, targets):
        # takes a list of tensors, data types need to be cleaned
        # throughout the program
        # assert type(outputs) is Tensor
        # assert type(targets) is Tensor
        total = 0
        for i in range(len(outputs)):
            total = total + (outputs[i].data-targets[i])**2
        return total / len(outputs)

