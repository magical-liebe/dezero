"""Functions for Dezero."""

import numpy as np
from nptyping import NDArray

from dezero import Function, Variable


class Sin(Function):
    """Sin function."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward method."""
        assert len(xs) == 1
        x = xs[0]
        y = np.sin(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        """Backward method."""
        assert len(self.inputs) == 1
        assert len(gys) == 1
        gy = gys[0]
        x = self.inputs[0]
        gx = gy * np.cos(x.data)
        return gx


def sin(x: Variable) -> Variable:
    """Sin function."""
    y = Sin()(x)
    assert not isinstance(y, list)
    return y
