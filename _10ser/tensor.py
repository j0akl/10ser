import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True):
        # add support for devices later
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self.prev = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def rank(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def rand(cls, *dims):
        return cls(np.random.rand(*dims))

    def matmul(self, w):
        assert type(w) is Tensor, "input must be tensor"
        return Tensor(np.dot(self.data, w.data))
