"""Error types for neurodes.

The error messages are a first-class feature, not a fallback. Someone learning what a
neural network *is* meets these constantly, and the difference between

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x784 and 128x10)

and

    Linear got [B, 28, 28] but needs the last dimension to be 128.
    Hint: add a Flatten before this node to turn [B, 28, 28] into [B, 784].

is the difference between a toy and a teacher.
"""

from __future__ import annotations


class NeurodesError(Exception):
    """Base for every error this package raises on purpose."""

    def __init__(self, message: str, hint: str | None = None):
        self.message = message
        self.hint = hint
        super().__init__(self.render())

    def render(self) -> str:
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ShapeError(NeurodesError):
    """A layer was handed a tensor it cannot consume."""


class ParseError(NeurodesError):
    """A widget value could not be read (a shape string, a dim list, ...)."""


class GraphError(NeurodesError):
    """The traced graph is malformed: missing input, disconnected output, ..."""


class BuildError(NeurodesError):
    """The graph is valid but could not be turned into a torch module."""
