"""Tests for core."""

from typing import Callable

import numpy as np
import pytest

from dezero import Function, Variable, as_array, no_grad
from dezero.core_simple import exp, square
from dezero.utils import get_dot_graph


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> float:
    """Numerical differentiation."""
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def numerical_diff_two(
    f: Callable[[Variable, Variable], Variable], x: Variable, y: Variable, eps: float = 1e-4
) -> tuple[float, float]:
    """Numerical differentiation for 2 inputs."""
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y = Variable(as_array(y.data))
    z0 = f(x0, y)
    z1 = f(x1, y)
    dx = (z1.data - z0.data) / (2 * eps)

    x = Variable(as_array(x.data))
    y0 = Variable(as_array(y.data - eps))
    y1 = Variable(as_array(y.data + eps))
    z0 = f(x, y0)
    z1 = f(x, y1)
    dy = (z1.data - z0.data) / (2 * eps)

    return dx, dy


class TestVariable:
    """Test class for Variable."""

    def test_init(self) -> None:
        """Test init."""
        x = Variable(np.array(0.5))
        assert x.data == np.array(0.5)
        assert x.grad is None
        assert x.creator is None

    def test_raise_type_error(self) -> None:
        """Test raise type error."""
        with pytest.raises(TypeError):
            Variable(0.5)  # type: ignore

    def test_property(self) -> None:
        """Test property."""
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        assert x.shape == (2, 3)
        assert x.ndim == 2
        assert x.size == 6
        assert x.dtype == np.int64

        assert len(x) == 2
        assert repr(x) == "variable([[1 2 3]\n          [4 5 6]])"
        assert repr(Variable(None)) == "variable(None)"


class TestFunction:
    """Test class for Function."""

    def test_raise_not_implemented_error(self) -> None:
        """Test raise not implemented error."""
        f = Function()
        with pytest.raises(NotImplementedError):
            f.forward()
        with pytest.raises(NotImplementedError):
            f.backward()

    def test_no_grad(self) -> None:
        """Test no_grad."""
        x = Variable(np.random.rand(1))
        t = square(x)
        y = square(t)
        y.backward()
        assert t.creator is not None
        assert t.generation == 1
        assert y.creator is not None
        assert y.generation == 2

        with no_grad():
            x.cleargrad()
            t = square(x)
            y = square(t)
            assert t.creator is None
            assert t.generation == 0
            assert y.creator is None
            assert y.generation == 0
            with pytest.raises(AttributeError):
                y.backward()


class TestSquare:
    """Test class for square."""

    def test_forward(self) -> None:
        """Test forward."""
        a = np.random.rand(1)

        x = Variable(a)
        y = square(x)

        expected = a**2
        assert y.data == expected

    def test_gradient_check(self) -> None:
        """Test gradient check."""
        a = np.random.rand(1)

        x = Variable(a)
        y = square(x)
        y.backward()

        num_grad = numerical_diff(square, x)
        assert np.allclose(x.grad, num_grad)


class TestExp:
    """Test class for exp."""

    def test_forward(self) -> None:
        """Test forward."""
        a = np.random.rand(1)

        x = Variable(a)
        y = exp(x)
        expected = np.array(np.exp(a))
        assert y.data == expected

    def test_gradient_check(self) -> None:
        """Test gradient check."""
        a = np.random.rand(1)

        x = Variable(np.array(a))
        y = exp(x)
        y.backward()

        num_grad = numerical_diff(exp, x)
        assert np.allclose(x.grad, num_grad)


class TestBasicCalculation:
    """Test class for basic calculation."""

    def test_add(self) -> None:
        """Test add."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        x = Variable(a)
        y = Variable(b)
        z = x + y
        z.backward()
        assert z.data == a + b
        assert x.grad == 1
        assert y.grad == 1

        x.cleargrad()
        z = x + 2
        z.backward()
        assert z.data == a + 2
        assert x.grad == 1

        x.cleargrad()
        z = x + 3.5
        z.backward()
        assert z.data == a + 3.5
        assert x.grad == 1

        x.cleargrad()
        z = 5 + x
        z.backward()
        assert z.data == 5 + a
        assert x.grad == 1

        x.cleargrad()
        z = 2.5 + x
        z.backward()
        assert z.data == 2.5 + a
        assert x.grad == 1

        x.cleargrad()
        y = x + x
        y.backward()
        assert x.grad == 2

        x.cleargrad()
        y = x + x + x
        y.backward()
        assert x.grad == 3

    def test_mul(self) -> None:
        """Test multiply."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        xa = Variable(a)
        xb = Variable(b)

        y = xa * xb
        y.backward()
        assert y.data == a * b
        assert xa.grad == b
        assert xb.grad == a

        xa.cleargrad()
        y = xa * 10
        y.backward()
        assert y.data == a * 10
        assert xa.grad == 10

        xa.cleargrad()
        y = 7.5 * xa
        y.backward()
        assert y.data == 7.5 * a
        assert xa.grad == 7.5

        xa.cleargrad()
        y = xa * xa
        y.backward()
        assert y.data == a * a
        assert xa.grad == 2 * a

    def test_neg(self) -> None:
        """Test negative."""
        a = np.random.rand(1)

        x = Variable(a)
        y = -x
        y.backward()
        assert y.data == -a
        assert x.grad == -1

    def test_sub(self) -> None:
        """Test subtract."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        xa = Variable(a)
        xb = Variable(b)
        y = xa - xb
        y.backward()
        assert y.data == a - b
        assert xa.grad == 1
        assert xb.grad == -1

        xa.cleargrad()
        xb.cleargrad()
        y = xb - xa
        y.backward()
        assert y.data == b - a
        assert xa.grad == -1
        assert xb.grad == 1

        xa.cleargrad()
        y = xa - 2
        y.backward()
        assert y.data == a - 2
        assert xa.grad == 1

        xa.cleargrad()
        y = 2.5 - xa
        y.backward()
        assert y.data == 2.5 - a
        assert xa.grad == -1

    def test_div(self) -> None:
        """Test divide."""
        a = np.array(2.5)
        b = np.array(4)

        xa = Variable(a)
        xb = Variable(b)
        y = xa / xb
        y.backward()
        assert y.data == a / b
        assert xa.grad == 1 / b
        assert xb.grad == -a / b**2

        xa.cleargrad()
        xb.cleargrad()
        y = xa / 2
        y.backward()
        assert y.data == a / 2
        assert xa.grad == 1 / 2

        xa.cleargrad()
        y = 7.5 / xa
        y.backward()
        assert y.data == 7.5 / a
        assert xa.grad == -7.5 / a**2

    def test_pow(self) -> None:
        """Test power."""
        a = np.random.rand(1)

        x = Variable(a)
        y = x**3
        y.backward()
        assert y.data == a**3
        assert x.grad == 3 * a**2


class TestComplexFucntion:
    """Test class for complex function."""

    def sphere(self, x: Variable, y: Variable) -> Variable:
        """Sphere function."""
        return x**2 + y**2

    def test_sphere(self) -> None:
        """Test sphere function."""
        a = np.random.rand(1)
        b = np.random.rand(1)
        x = Variable(a)
        y = Variable(b)

        z = self.sphere(x, y)
        assert z.data == a**2 + b**2

        z.backward()
        assert x.grad == 2 * a
        assert y.grad == 2 * b

    def matyas(self, x: Variable, y: Variable) -> Variable:
        """Matyas function."""
        return 0.26 * (x**2 + y**2) - 0.48 * x * y

    def test_matyas(self) -> None:
        """Test matyas function."""
        a = np.random.rand(1)
        b = np.random.rand(1)
        x = Variable(a)
        y = Variable(b)

        z = self.matyas(x, y)
        assert z.data == 0.26 * (a**2 + b**2) - 0.48 * a * b

        z.backward()
        assert x.grad == 2 * a * 0.26 - 0.48 * b
        assert y.grad == 2 * b * 0.26 - 0.48 * a

    def goldstein_price(self, x: Variable, y: Variable) -> Variable:
        """Goldstein-Price function."""
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
        )

    def test_goldstein_price(self) -> None:
        """Test goldstein-price function."""
        a = np.array(1.0)
        b = np.array(1.0)
        x = Variable(a)
        x.name = "x"
        y = Variable(b)
        y.name = "y"

        z = self.goldstein_price(x, y)
        z.name = "z"
        z.backward()
        num_grad_x, num_grad_y = numerical_diff_two(self.goldstein_price, x, y)
        assert np.allclose(x.grad, num_grad_x)
        assert np.allclose(y.grad, num_grad_y)

        get_dot_graph(z, verbose=True)
