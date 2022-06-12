"""DeZero: Deep Learning framework for Python."""

from pathlib import Path

from single_source import get_version

__version__ = get_version(__name__, Path(__file__).parent)

from dezero import functions
from dezero.core import Function, Variable, as_array, as_variable, no_grad, using_config

__all__ = ["Function", "Variable", "as_array", "as_variable", "no_grad", "using_config", "functions"]
