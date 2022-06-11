"""Tests for functions."""

import numpy as np

from dezero import Variable
from dezero.functions import sin


class TestFunctions:
    """Tests for functions."""

    def test_sin(self):
        """Test sin function."""
        a = np.array(np.pi / 4)

        x = Variable(a)
        y = sin(x)
        y.backward()

        assert y.data == np.sin(a)
        assert x.grad == np.cos(a)
