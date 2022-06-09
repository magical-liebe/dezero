"""Core module of DeZero."""

from nptyping import NDArray


class Variable:
    """Variable class."""

    def __init__(self, data: NDArray) -> None:
        """Initialize Variable class."""
        self.data = data
        self.grad: NDArray | None = None
        self.creator: Function | None = None

    def set_creator(self, func: "Function") -> None:
        """Set creator function."""
        self.creator = func

    def backward(self) -> None:
        """Backward propagation."""
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            if f is not None:
                x, y = f.input, f.output
                if y.grad is not None:
                    x.grad = f.backward(y.grad)

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    """Function class."""

    def __call__(self, input_v: Variable) -> Variable:
        """Call function."""
        x = input_v.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input_v
        self.output = output
        return output

    def forward(self, x: NDArray) -> NDArray:
        """Forward propagation."""
        raise NotImplementedError()

    def backward(self, gy: NDArray) -> NDArray:
        """Backward propagation."""
        raise NotImplementedError()
