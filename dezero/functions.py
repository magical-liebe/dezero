"""Functions for Dezero."""

import numpy as np
from nptyping import NDArray

from dezero.core import Function, Variable


class Square(Function):
    """Square function class."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        return xs[0] ** 2

    def backward(self, *gys: NDArray) -> NDArray:
        """Backward propagation."""
        x = self.inputs[0].data
        gx = 2 * x * gys[0]
        return gx


class Exp(Function):
    """Exponential function class."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        return np.exp(xs[0])

    def backward(self, *gys: NDArray) -> NDArray:
        """Backward propagation."""
        x = self.inputs[0].data
        gx = np.exp(x) * gys[0]
        return gx


class Add(Function):
    """Add function class."""

    def forward(self, *xs: NDArray) -> NDArray | list[NDArray]:
        """Forward propagation."""
        x0, x1 = xs
        y = x0 + x1
        return y

    def backward(self, *gys: NDArray) -> list[NDArray]:
        """Backward propagation."""
        return [gys[0], gys[0]]


def square(x: Variable) -> Variable:
    """Square function."""
    y = Square()(x)
    assert not isinstance(y, list)
    return y


def exp(x: Variable) -> Variable:
    """Exponential function."""
    y = Exp()(x)
    assert not isinstance(y, list)
    return y


def add(x0: Variable, x1: Variable) -> Variable:
    """Add function."""
    y = Add()(x0, x1)
    assert not isinstance(y, list)
    return y
