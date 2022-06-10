"""Core module of DeZero."""

import weakref

import numpy as np
from nptyping import NDArray

from dezero.config import Config


class Variable:
    """Variable class."""

    def __init__(self, data: NDArray, name: str = None) -> None:
        """Initialize Variable class."""
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad: NDArray | None = None
        self.creator: Function | None = None
        self.generation: int = 0

    def set_creator(self, func: "Function") -> None:
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

    # def __mul__(self, other: "Variable") -> "Variable":
    #     """Multiply."""
    #     return mul(self, other)


class Function:
    """Function class."""

    def __call__(self, *inputs: Variable) -> Variable | list[Variable]:
        """Call function."""
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, list):
            ys = [ys]
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: NDArray) -> NDArray | list[NDArray]:
        """Forward propagation."""
        raise NotImplementedError()

    def backward(self, *gys: NDArray) -> NDArray | list[NDArray]:
        """Backward propagation."""
        raise NotImplementedError()


def as_array(x: NDArray) -> NDArray:
    """Convert to numpy array."""
    if np.isscalar(x):
        return np.array(x)
    return x
