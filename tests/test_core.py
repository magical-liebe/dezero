"""Tests for core."""

from typing import Callable

import numpy as np
import pytest

from dezero.config import no_grad
from dezero.core import Function, Variable, as_array, exp, square


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> float:
    """Numerical differentiation."""
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


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


class TestAdd:
    """Test class for add."""

    def test_forward(self) -> None:
        """Test forward."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        x = Variable(a)
        y = Variable(b)
        z = x + y
        expected = np.array(a + b)
        assert z.data == expected

    def test_backward(self) -> None:
        """Test backward."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        x = Variable(a)
        y = Variable(b)
        z = x + y
        z.backward()
        expected = np.array(1)
        assert x.grad == expected
        assert y.grad == expected

        x.cleargrad()
        y = x + x
        y.backward()
        assert x.grad == 2 * expected

        x.cleargrad()
        y = x + x + x
        y.backward()
        assert x.grad == 3 * expected


class TestMul:
    """Test class for multiply."""

    def test_forwart_backward(self) -> None:
        """Test forward and backward."""
        a = np.random.rand(1)
        b = np.random.rand(1)
        c = np.random.rand(1)

        xa = Variable(a)
        xb = Variable(b)
        xc = Variable(c)

        y = xa * xb + xc
        assert y.data == np.array(a * b + c)

        y.backward()
        assert xa.grad == np.array(b)
        assert xb.grad == np.array(a)
        assert xc.grad == np.array(1)


class TestNeg:
    """Test class for neg."""

    def test_forward_backward(self) -> None:
        """Test forward and backward."""
        a = np.random.rand(1)

        x = Variable(a)
        y = -x
        expected = -a
        assert y.data == expected

        y.backward()
        assert x.grad == -1


class TestSub:
    """Test class for subtract."""

    def test_forward_backwart(self) -> None:
        """Test forward and backward."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        x = Variable(a)
        y = Variable(b)
        z = x - y
        expected = a - b
        assert z.data == expected

        z.backward()
        assert x.grad == 1
        assert y.grad == -1

        z = y - x
        expected = b - a
        assert z.data == expected


class TestDiv:
    """Test class for divide."""

    def test_forward_backward(self) -> None:
        """Test forward and backward."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        x = Variable(a)
        y = Variable(b)
        z = x / y
        expected = a / b
        assert z.data == expected

        z.backward()
        assert x.grad == 1 / b
        assert y.grad == -a / b**2


class TestPow:
    """Test class for power."""

    def test_forward_backward(self) -> None:
        """Test forward and backward."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        x = Variable(a)
        y = Variable(b)
        z = x**y
        expected = a**b
        assert z.data == expected

        z.backward()
        assert x.grad == b * a ** (b - 1)
        assert y.grad == a**b * np.log(a)


def test_chain() -> None:
    """Test chain."""
    a = 0.5

    x = Variable(np.array(a))
    y = square(exp(square(x)))
    expected = np.array(np.exp(a**2.0) ** 2.0)
    assert y.data == expected

    y.backward()
    expected = np.array(4 * a * np.exp(a**2) ** 2)
    assert x.grad == expected


def test_fork() -> None:
    """Test forked graph."""
    k = 2.0

    x = Variable(np.array(k))
    a = square(x)
    y = square(a) + square(a)
    y.backward()

    assert y.data == 32.0
    assert x.grad == 64.0


def test_disable_backprop() -> None:
    """Test disabling backprop."""
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


def test_different_type() -> None:
    """Test different type."""
    a = np.random.rand(1)
    x = Variable(a)
    y = np.random.rand(1)

    z = x + y
    assert z.data == a + y

    z = x * y
    assert z.data == a * y

    z = y + x
    assert z.data == a + y

    z = y * x
    assert z.data == a * y

    z = x + 5
    assert z.data == a + np.array(5)

    z = 3 + x
    assert z.data == a + np.array(3)

    z = x * 5
    assert z.data == a * np.array(5)

    z = 3 * x
    assert z.data == a * np.array(3)

    z = x - 5
    assert z.data == a - np.array(5)

    z = 3 - x
    assert z.data == np.array(3) - a

    z = x / 5
    assert z.data == a / np.array(5)

    z = 3 / x
    assert z.data == np.array(3) / a

    z = x**5
    assert z.data == a**5

    z = 3**x
    assert z.data == 3**a
