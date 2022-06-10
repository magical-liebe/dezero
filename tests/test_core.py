"""Tests for core."""

import numpy as np
import pytest

from dezero.core import Function, Variable


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
