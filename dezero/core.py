"""Core module of DeZero."""

from __future__ import annotations

import weakref

import numpy as np
from nptyping import NDArray

from dezero.config import Config


class Variable:
    """Variable class."""

    __array_priority__ = 200

    def __init__(self, data: NDArray, name: str = None) -> None:
        """Initialize Variable class."""
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad: NDArray | None = None
        self.creator: Function | None = None
        self.generation: int = 0

    def set_creator(self, func: Function) -> None:
        """Set creator function."""
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self) -> None:
        """Clear gradient."""
        self.grad = None

    def backward(self, retain_grad: bool = False) -> None:
        """Backward propagation."""
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = []
        seen_set = set()

        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, list):
                gxs = [gxs]

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Get ndim."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Get size."""
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        """Get dtype."""
        return self.data.dtype

    def __len__(self) -> int:
        """Get length."""
        return len(self.data)

    def __repr__(self) -> str:
        """Get representation."""
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def __add__(self, other: Variable | NDArray | int | float) -> Variable:
        """Add."""
        return add(self, other)

    def __radd__(self, other: Variable | NDArray | int | float) -> Variable:
        """Add for right."""
        print(type(other))
        return add(self, other)

    def __mul__(self, other: Variable | NDArray | int | float) -> Variable:
        """Multiply."""
        return mul(self, other)

    def __rmul__(self, other: Variable | NDArray | int | float) -> Variable:
        """Multiply for right."""
        return mul(self, other)

    def __neg__(self) -> Variable:
        """Negate."""
        return neg(self)

    def __sub__(self, other: Variable | NDArray | int | float) -> Variable:
        """Subtract."""
        return sub(self, other)

    def __rsub__(self, other: Variable | NDArray | int | float) -> Variable:
        """Subtract for right."""
        return rsub(self, other)

    def __truediv__(self, other: Variable | NDArray | int | float) -> Variable:
        """Divide."""
        return div(self, other)

    def __rtruediv__(self, other: Variable | NDArray | int | float) -> Variable:
        """Divide for right."""
        return rdiv(self, other)

    def __pow__(self, other: Variable | NDArray | int | float) -> Variable:
        """Power."""
        return lpow(self, other)

    def __rpow__(self, other: Variable | NDArray | int | float) -> Variable:
        """Power for right."""
        return rpow(self, other)


class Function:
    """Function class."""

    def __call__(self, *inputs: Variable | NDArray) -> Variable | list[Variable]:
        """Call function."""
        inputs_v = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs_v]
        ys = self.forward(*xs)
        if not isinstance(ys, list):
            ys = [ys]
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs_v])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs_v
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: NDArray) -> NDArray | list[NDArray]:
        """Forward propagation."""
        raise NotImplementedError()

    def backward(self, *gys: NDArray) -> NDArray | list[NDArray]:
        """Backward propagation."""
        raise NotImplementedError()


def as_array(x: NDArray | int | float) -> NDArray:
    """Convert to numpy array."""
    if np.isscalar(x):
        return np.array(x)
    assert isinstance(x, NDArray)
    return x


def as_variable(obj: Variable | NDArray) -> Variable:
    """Convert to variable."""
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


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

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        assert len(xs) == 2
        x0, x1 = xs
        y = x0 + x1
        return y

    def backward(self, *gys: NDArray) -> list[NDArray]:
        """Backward propagation."""
        assert len(gys) == 1
        return [gys[0], gys[0]]


class Mul(Function):
    """Multiply function class."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        assert len(xs) == 2
        x0, x1 = xs
        y = x0 * x1
        return y

    def backward(self, *gys: NDArray) -> list[NDArray]:
        """Backward propagation."""
        assert len(gys) == 1
        gy = gys[0]
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return [gy * x1, gy * x0]


class Neg(Function):
    """Negative function class."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        assert len(xs) == 1
        x = xs[0]
        return -x

    def backward(self, *gys: NDArray) -> NDArray:
        """Backward propagation."""
        assert len(gys) == 1
        gy = gys[0]
        return -gy


class Sub(Function):
    """Subtract function class."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        assert len(xs) == 2
        x0, x1 = xs
        y = x0 - x1
        return y

    def backward(self, *gys: NDArray) -> list[NDArray]:
        """Backward propagation."""
        assert len(gys) == 1
        return [gys[0], -gys[0]]


class Div(Function):
    """Divide function class."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        assert len(xs) == 2
        x0, x1 = xs
        y = x0 / x1
        return y

    def backward(self, *gys: NDArray) -> list[NDArray]:
        """Backward propagation."""
        assert len(gys) == 1
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gy = gys[0]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return [gx0, gx1]


class Pow(Function):
    """Power function class."""

    def forward(self, *xs: NDArray) -> NDArray:
        """Forward propagation."""
        assert len(xs) == 2
        x0, x1 = xs
        y = x0**x1
        return y

    def backward(self, *gys: NDArray) -> list[NDArray]:
        """Backward propagation."""
        assert len(gys) == 1
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gy = gys[0]
        gx0 = x1 * x0 ** (x1 - 1) * gy
        gx1 = np.log(x0) * x0**x1 * gy
        return [gx0, gx1]


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


def add(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Add function."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Add()(x0, x1)
    assert not isinstance(y, list)
    return y


def mul(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Multiply function."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Mul()(x0, x1)
    assert not isinstance(y, list)
    return y


def neg(x: Variable) -> Variable:
    """Negative function."""
    y = Neg()(x)
    assert not isinstance(y, list)
    return y


def sub(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Subtract function."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Sub()(x0, x1)
    assert not isinstance(y, list)
    return y


def rsub(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Subtract function for right."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Sub()(x1, x0)
    assert not isinstance(y, list)
    return y


def div(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Divide function."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Div()(x0, x1)
    assert not isinstance(y, list)
    return y


def rdiv(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Divide function for right."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Div()(x1, x0)
    assert not isinstance(y, list)
    return y


def lpow(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Power function."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Pow()(x0, x1)
    assert not isinstance(y, list)
    return y


def rpow(x0: Variable, x1: Variable | NDArray | int | float) -> Variable:
    """Power function for right."""
    if not isinstance(x1, Variable):
        x1 = as_array(x1)
    y = Pow()(x1, x0)
    assert not isinstance(y, list)
    return y
