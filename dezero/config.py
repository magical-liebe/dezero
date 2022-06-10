"""Configuration class for dezero."""

import contextlib
from typing import Any, Generator


class Config:
    """Configuration class."""

    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: Any) -> Generator:
    """Context manager to change config."""
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad() -> contextlib._GeneratorContextManager:
    """Context manager to disable grad."""
    return using_config("enable_backprop", False)
