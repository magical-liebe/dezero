import numpy as np
from nptyping import NDArray

from variable import Variable


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: NDArray) -> NDArray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: NDArray) -> NDArray:
        return x**2


class Exp(Function):
    def forward(self, x: NDArray) -> NDArray:
        return np.exp(x)


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    print(y.data)
