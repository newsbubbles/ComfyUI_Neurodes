"""Helpers shared by the layer definition modules."""

from __future__ import annotations

from typing import Any

from ..shape import Dim, Shape
from ..trace import SymTensor


def out(shape: Shape, dtype: str = "float32") -> SymTensor:
    """Build the SymTensor an ``infer`` function returns (producer is filled in later)."""
    return SymTensor(shape=shape, dtype=dtype)


def same(t: SymTensor) -> SymTensor:
    """Shape- and dtype-preserving layer."""
    return out(t.shape, t.dtype)


def fmt_num(v: Any) -> str:
    """Format a value for emitted Python source."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, tuple):
        return "(" + ", ".join(fmt_num(x) for x in v) + ")"
    return repr(v)


def kwargs_src(**kw) -> str:
    """``bias=False, stride=2`` — omitting nothing, because explicit reads better."""
    return ", ".join(f"{k}={fmt_num(v)}" for k, v in kw.items())


def dim_or_symbol(d: Dim, fallback_name: str) -> Dim:
    return d if d.is_concrete else Dim(fallback_name)
