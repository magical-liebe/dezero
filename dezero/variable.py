import numpy as np
from nptyping import NDArray


class Variable:
    def __init__(self, data: NDArray) -> None:
        self.data = data


if __name__ == "__main__":
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
