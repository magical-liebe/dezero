"""Functions for Dezero."""

import numpy as np
from nptyping import NDArray

from dezero.core import Function, Variable


class Square(Function):
    """Square function class."""

    def forward(self, x: NDArray) -> NDArray:
        """Forward propagation."""
        return x**2

    def backward(self, gy: NDArray) -> NDArray:
        """Backward propagation."""
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    """Exponential function class."""

    def forward(self, x: NDArray) -> NDArray:
        """Forward propagation."""
        return np.exp(x)

    def backward(self, gy: NDArray) -> NDArray:
        """Backward propagation."""
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: Variable) -> Variable:
    """Square function."""
    return Square()(x)


def exp(x: Variable) -> Variable:
    """Exponential function."""
    return Exp()(x)


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)
