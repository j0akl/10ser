import unittest
from _10ser.tensor import Tensor
import _10ser.layer as L

class LayerTest(unittest.TestCase):
    def test_sigmoid_output(self):
        data = 0.1
        s = L.Sigmoid()
        result = s(data)
        self.assertEqual(result, 0.52497918747894)

    def test_relu_output(self):
        data  = [-1., 0, 1., 2.]
        results = []
        r = L.ReLU()
        for i in data:
            results.append(r(Tensor(i)))
        self.assertListEqual(results, [0., 0., 1., 2.,])

    def test_linear_given_weights(self):
        x = [Tensor([i]) for i in range(10)]
        y = [[i] for i in range(10)]
        net = L.Linear(1, 1)
        net.weights = Tensor([1])
        net.biases = Tensor([0])
        result = []
        for i in range(len(x)):
            result.append(net(x[i]))

        self.assertListEqual(y, result)

