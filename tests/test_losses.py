import unittest
from _10ser.tensor import Tensor
from _10ser.layer import Linear
from _10ser.loss import MSELoss

class LossTest(unittest.TestCase):
    def test_mse_loss(self):
        data = [Tensor([i]) for i in range(10)]
        targets = [Tensor([i]) for i in range(10)]
        net = Linear(1, 1)
        net.weights = Tensor([1])
        net.biases = Tensor([0])
        loss_fn = MSELoss()
        loss = loss_fn([net(i) for i in data], targets)
        self.assertEqual(loss, [0])

    def test_mse_loss_nonzero(self):
        data = [Tensor([i + 1]) for i in range(10)]
        targets = [Tensor([i]) for i in range(10)]
        net = Linear(1, 1)
        net.weights = Tensor([1])
        net.biases = Tensor([0])
        loss_fn = MSELoss()
        loss = loss_fn([net(i) for i in data], targets)
        self.assertEqual(loss, [1])

