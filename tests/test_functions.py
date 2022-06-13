"""Tests for functions."""

import numpy as np

import dezero.functions as F
from dezero import Variable


class TestFunctions:
    """Tests for functions."""

    def test_sin(self) -> None:
        """Test sin function."""
        a = np.array(np.pi / 4)

        x = Variable(a)
        y = F.sin(x)
        y.backward(create_graph=True)
        assert y.data == np.sin(a)
        assert x.grad.data == np.cos(a)

        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        assert x.grad.data == -np.sin(a)

        gx2 = x.grad
        x.cleargrad()
        gx2.backward(create_graph=True)
        assert x.grad.data == -np.cos(a)

        gx3 = x.grad
        x.cleargrad()
        gx3.backward(create_graph=True)
        assert x.grad.data == np.sin(a)

    def test_cos(self) -> None:
        """Test cos function."""
        a = np.array(np.pi / 4)

        x = Variable(a)
        y = F.cos(x)
        y.backward()
        assert y.data == np.cos(a)
        assert x.grad.data == -np.sin(a)

    def test_tanh(self) -> None:
        """Test tanh function."""
        a = np.array(1.0)

        x = Variable(a)
        x.name = "x"
        y = F.tanh(x)
        y.name = "y"
        y.backward(create_graph=True)
        assert y.data == np.tanh(a)
        assert x.grad.data == 1 - np.tanh(a) ** 2

    def test_reshape(self) -> None:
        """Test reshape function."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        x = Variable(a)
        y = F.reshape(x, (6,))
        y.backward()
        assert y.shape == (6,)
        assert x.grad.shape == x.shape
        assert np.allclose(x.grad.data, np.ones_like(x.data))

        y = F.reshape(a, (2, 3))
        assert y.shape == (2, 3)
        assert np.allclose(y.data, a)

        x = Variable(np.random.randn(1, 2, 3))
        y = x.reshape((2, 3))
        assert y.shape == (2, 3)

        y = x.reshape([3, 2])
        assert y.shape == (3, 2)

    def test_transpose(self) -> None:
        """Test transpose function."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        x = Variable(a)
        y = F.transpose(x)
        y.backward()
        assert y.shape == (3, 2)
        assert x.grad.shape == x.shape
        assert np.allclose(x.grad.data, np.ones_like(x.data))

        x.cleargrad()
        y = x.T
        y.backward()
        assert y.shape == (3, 2)
        assert x.grad.shape == x.shape
        assert np.allclose(x.grad.data, np.ones_like(x.data))

        x = Variable(np.random.rand(1, 2, 3, 4))
        y = F.transpose(x, (1, 0, 3, 2))
        y.backward()
        assert y.shape == (2, 1, 4, 3)
        assert x.grad.shape == x.shape

    def test_sum(self) -> None:
        """Test sum function."""
        a = np.array([1, 2, 3, 4, 5, 6])
        x = Variable(a)
        y = F.sum(x)
        # y.backward()
        assert y.data == np.sum(a)
        # assert np.allclose(x.grad.data, np.ones_like(x.data))

        a = np.array([[1, 2, 3], [4, 5, 6]])
        x = Variable(a)
        y = F.sum(x)
        # y.backward()
        assert y.data == np.sum(a)
        # assert np.allclose(x.grad.data, np.ones_like(x.data))
