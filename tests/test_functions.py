"""Tests for functions."""

from typing import Callable

import numpy as np
import pytest

from dezero.config import no_grad
from dezero.core import Variable, as_array
from dezero.functions import add, exp, mul, square


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
        z = add(x, y)
        expected = np.array(a + b)
        assert z.data == expected

    def test_backward(self) -> None:
        """Test backward."""
        a = np.random.rand(1)
        b = np.random.rand(1)

        x = Variable(a)
        y = Variable(b)
        z = add(x, y)
        z.backward()
        expected = np.array(1)
        assert x.grad == expected
        assert y.grad == expected

        x.cleargrad()
        y = add(x, x)
        y.backward()
        assert x.grad == 2 * expected

        x.cleargrad()
        y = add(add(x, x), x)
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

        y = add(mul(xa, xb), xc)
        assert y.data == np.array(a * b + c)

        y.backward()
        assert xa.grad == np.array(b)
        assert xb.grad == np.array(a)
        assert xc.grad == np.array(1)


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
    y = add(square(a), square(a))
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


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> float:
    """Numerical differentiation."""
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
