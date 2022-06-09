"""Core module of DeZero."""

import numpy as np
from nptyping import NDArray


class Variable:
    """Variable class."""

    def __init__(self, data: NDArray) -> None:
        """Initialize Variable class."""
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError("{} is not supported".format(type(data)))

        self.data = data
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

    def backward(self) -> None:
        """Backward propagation."""
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            if f is not None:
                gys = [output.grad for output in f.outputs]
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


class Function:
    """Function class."""

    def __call__(self, *inputs: Variable) -> Variable | list[Variable]:
        """Call function."""
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, list):
            ys = [ys]
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

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
