"""Tests for functions."""

from typing import Callable

import numpy as np

from dezero.core import Variable, as_array
from dezero.functions import exp, square


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


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> float:
    """Numerical differentiation."""
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
