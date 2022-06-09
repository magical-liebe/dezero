"""Functions for Dezero."""

import numpy as np
from nptyping import NDArray

from dezero.core import Function, Variable


class Square(Function):
    """Square function."""

    def forward(self, x: NDArray) -> NDArray:
        """Forward propagation."""
        return x**2

    def backward(self, gy: NDArray) -> NDArray:
        """Backward propagation."""
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """Exponential function."""

    def forward(self, x: NDArray) -> NDArray:
        """Forward propagation."""
        return np.exp(x)

    def backward(self, gy: NDArray) -> NDArray:
        """Backward propagation."""
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
