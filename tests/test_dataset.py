import unittest
from _10ser.tensor import Tensor
from _10ser.dataset import Dataset

class DatasetTest(unittest.TestCase):
    def test_creation_no_shuffle_or_split(self):
        x = Tensor([1, 2, 3])
        y = Tensor([5, 6, 7])
        expected = Tensor([(1, 5), (2, 6), (3, 7)])
        ds = Dataset(x, y)
        result = ds.create(shuffle=False)
        self.assertTrue((result.data == expected.data).all())

    def test_creation_no_shuffle(self):
        x = Tensor([1, 2, 3])
        y = Tensor([5, 6, 7])
        expected = Tensor([(1, 5), (2, 6)])
        ds = Dataset(x, y)
        result, _ = ds.create(shuffle=False, split=0.8)
        self.assertTrue((result.data == expected.data).all())
