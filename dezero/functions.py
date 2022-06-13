"""Functions for Dezero."""

import numpy as np
from nptyping import NDArray

from dezero import utils
from dezero.core import Function, Variable, as_variable


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
        gx = gy * cos(x)
        return gx


class Cos(Function):
    """Cos function."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward method."""
        assert len(xs) == 1
        x = xs[0]
        y = np.cos(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        """Backward method."""
        assert len(self.inputs) == 1
        assert len(gys) == 1
        gy = gys[0]
        x = self.inputs[0]
        gx = gy * -sin(x)
        return gx


class Tanh(Function):
    """Tanh function."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward method."""
        assert len(xs) == 1
        x = xs[0]
        y = np.tanh(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        """Backward method."""
        assert len(self.outputs) == 1
        assert len(gys) == 1
        gy = gys[0]
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


class Reshape(Function):
    """Reshape function."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize."""
        self.shape = shape

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward method."""
        assert len(xs) == 1
        x = xs[0]
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, *gys: Variable) -> Variable:
        """Backward method."""
        assert len(gys) == 1
        gy = gys[0]
        return reshape(gy, self.x_shape)


class Transpose(Function):
    """Transpose function."""

    def __init__(self, axes: tuple[int, ...] = None) -> None:
        """Initialize."""
        self.axes = axes

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward method."""
        assert len(xs) == 1
        x = xs[0]
        y = x.transpose(self.axes)
        return y

    def backward(self, *gys: Variable) -> Variable:
        """Backward method."""
        assert len(gys) == 1
        gy = gys[0]
        if self.axes is None:
            return transpose(gy)
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


class Sum(Function):
    """Sum function."""

    def __init__(self, axis: int, keepdims: bool) -> None:
        """Initialize."""
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward method."""
        assert len(xs) == 1
        x = xs[0]
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, *gys: Variable) -> Variable:
        """Backward method."""
        assert len(gys) == 1
        gy = gys[0]
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sin(x: Variable | NDArray) -> Variable:
    """Sin function."""
    y = Sin()(x)
    assert not isinstance(y, list)
    return y


def cos(x: Variable | NDArray) -> Variable:
    """Cos function."""
    y = Cos()(x)
    assert not isinstance(y, list)
    return y


def tanh(x: Variable | NDArray) -> Variable:
    """Tanh function."""
    y = Tanh()(x)
    assert not isinstance(y, list)
    return y


def reshape(x: Variable | NDArray, shape: tuple[int, ...]) -> Variable:
    """Reshape function."""
    if x.shape == shape:
        return as_variable(x)
    y = Reshape(shape)(x)
    assert not isinstance(y, list)
    return y


def transpose(x: Variable | NDArray, axes: tuple[int, ...] = None) -> Variable:
    """Transpose function."""
    y = Transpose(axes)(x)
    assert not isinstance(y, list)
    return y


def sum(x: Variable | NDArray, axis: int = None, keepdims: bool = False) -> Variable:
    """Sum function."""
    y = Sum(axis, keepdims)(x)
    assert not isinstance(y, list)
    return y


def broadcast_to(x: Variable | NDArray, shape: tuple[int, ...]) -> Variable:
    """Broadcast to target shape."""
    raise NotImplementedError
