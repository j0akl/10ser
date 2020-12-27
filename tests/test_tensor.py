import unittest
import numpy as np
from _10ser.tensor import Tensor

class TensorTest(unittest.TestCase):
    def test_1d_input_int(self):
        data = [1, 2, 3]
        t = Tensor(data)
        self.assertListEqual(list(t.data), data)

    def test_2d_input_int(self):
        data = [[1, 2, 3], [3, 4, 5]]
        t = Tensor(data)
        self.assertTrue((np.array(data) == t.data).all())

    def test_1d_input_int(self):
        data = [1, 2, 3]
        t = Tensor(data)
        self.assertTrue((np.array(data) == t.data).all())

    def test_1d_from_nupmy(self):
        data = np.array([1, 2, 3])
        t = Tensor(data)
        self.assertTrue((data == t.data).all())

    def test_1d_shape(self):
        data = [1, 2, 3]
        t = Tensor(data)
        self.assertEqual(t.shape, (3,))

    def test_dtype_int(self):
        data = [1, 2, 3]
        t = Tensor(data)
        self.assertEqual(t.dtype, np.int)

    def test_dtype_float(self):
        data = [1., 2., 3.]
        t = Tensor(data)
        self.assertEqual(t.dtype, np.float64)

    def test_rand_init(self):
        t = Tensor.rand(4, 5)
        self.assertEqual(t.shape, (4, 5,))

    def test_matmul_not_tensor_input(self):
        inputs = Tensor([1, 2, 3])
        weights = [3, 2, 1]
        self.assertRaises(AssertionError, lambda: inputs.matmul(weights))

    def test_matmul(self):
        inputs = Tensor([1, 2, 3])
        weights = Tensor([3, 2, 1])
        out = inputs.matmul(weights)
        self.assertEqual(Tensor(10).data, out.data)
