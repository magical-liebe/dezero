"""Functions for Dezero."""

import numpy as np
from nptyping import NDArray

from dezero import Function
from dezero.core_simple import Variable


class Sin(Function):
    """Sin function."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward method."""
        assert len(xs) == 1
        x = xs[0]
        y = np.sin(x)
        return y

    def backward(self, *gys: NDArray) -> NDArray:
        """Backward method."""
        assert len(gys) == 1
        gy = gys[0]
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x: Variable) -> Variable:
    """Sin function."""
    y = Sin()(x)
    assert not isinstance(y, list)
    return y
