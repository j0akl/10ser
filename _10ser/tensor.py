import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True):
        # add support for devices later
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self.prev = None

    def __str__(self):
        return 'Tensor({} requires_grad={})'.format(self.data, self.requires_grad)

    def __repr__(self):
        return "<Tensor {} with grad={}>".format(self.data, self.requires_grad)

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
