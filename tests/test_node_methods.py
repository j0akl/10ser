import unittest
import numpy as np

from node import Node

class NodeTest(unittest.TestCase):

    def test_get_value(self):
        data = np.array([2, 3.4])
        weights = np.array([.1, .5])
        bias = 0.1
        node = Node(data)
        node.update_bias(bias)
        node.update_weights(weights)
        result = node.get_value()
        self.assertEqual(result, 2)

if __name__ == "__main__":
    unittest.main()


