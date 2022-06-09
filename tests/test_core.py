"""Tests for core."""

import numpy as np
import pytest

from dezero.core import Variable


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
