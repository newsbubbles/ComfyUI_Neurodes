"""Operators — the level below layers.

These have no weights. They are the arithmetic the diagram usually draws as a small circle
with a plus in it: residual connections, gating, scaling, attention scores. Having them as
first-class nodes is what makes the pack able to express an architecture rather than just
a stack of blocks.
"""

from __future__ import annotations

import re

import torch

from ..errors import ShapeError
from ..registry import P, VARIADIC, layer
from ..shape import Dim, Shape, broadcast, unify_dim
from ..trace import DTYPES, FLOAT_DTYPES
from ._common import out, same

CAT = "ops/math"


# ---------------------------------------------------------------------------
# Elementwise arithmetic
# ---------------------------------------------------------------------------

def _elementwise(key: str, display: str, symbol: str, fn, doc: str,
                 arity: int = VARIADIC, aliases=()):
    def infer(ins, cfg):
        dtypes = {t.dtype for t in ins}
        if len(dtypes) > 1:
            raise ShapeError(
                "the inputs have different dtypes: " + ", ".join(sorted(dtypes)) + ".",
                hint="Put a Cast on one of them so both match.",
            )
        shape = broadcast([t.shape for t in ins], what="the inputs to " + display)
        return out(shape, ins[0].dtype)

    def apply(m, ts, c, _fn=fn):
        acc = ts[0]
        for t in ts[1:]:
            acc = _fn(acc, t)
        return acc

    layer(
        key, display, CAT, doc=doc, aliases=aliases, arity=arity,
        build=None, apply=apply,
        emit_call=lambda attr, args, c, _s=symbol: "(" + f" {_s} ".join(args) + ")",
        trains=False,
    )(infer)


_elementwise(
    "add", "Add", "+", lambda a, b: a + b,
    "Adds tensors together, position by position. This is the residual or skip connection: "
    "letting the original signal past a block untouched is what makes very deep networks "
    "trainable at all.",
    aliases=("residual", "skip connection", "sum", "plus", "shortcut"))

_elementwise(
    "subtract", "Subtract", "-", lambda a, b: a - b,
    "Subtracts the second tensor from the first, position by position.",
    arity=2, aliases=("minus", "difference"))

_elementwise(
    "multiply", "Multiply", "*", lambda a, b: a * b,
    "Multiplies tensors position by position. A gate: multiplying by something between 0 "
    "and 1 lets the network decide how much of a signal to let through.",
    aliases=("gate", "elementwise product", "hadamard", "mask", "attention"))

_elementwise(
    "divide", "Divide", "/", lambda a, b: a / b,
    "Divides the first tensor by the second, position by position.",
    arity=2, aliases=("ratio",))


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------

layer(
    "scale", "Scale", CAT,
    doc="Multiplies everything by a fixed number. Attention divides its scores by the square "
        "root of the head width using exactly this.",
    aliases=("multiply by constant", "gain", "temperature"),
    params=(P("value", "float", 1.0, "The constant to multiply by.", min=-1e6, max=1e6, step=0.01),),
    build=None,
    apply=lambda m, ts, c: ts[0] * float(c["value"]),
    emit_call=lambda attr, args, c: f"({args[0]} * {float(c['value'])})",
    trains=False,
)(lambda ins, cfg: same(ins[0]))


layer(
    "offset", "Offset", CAT,
    doc="Adds a fixed number to everything.",
    aliases=("add constant", "bias"),
    params=(P("value", "float", 0.0, "The constant to add.", min=-1e6, max=1e6, step=0.01),),
    build=None,
    apply=lambda m, ts, c: ts[0] + float(c["value"]),
    emit_call=lambda attr, args, c: f"({args[0]} + {float(c['value'])})",
    trains=False,
)(lambda ins, cfg: same(ins[0]))


layer(
    "clamp", "Clamp", CAT,
    doc="Squeezes every value into a range, cutting off anything outside it.",
    aliases=("clip", "limit"),
    params=(
        P("min", "float", 0.0, "Lowest value allowed.", min=-1e6, max=1e6, step=0.01),
        P("max", "float", 1.0, "Highest value allowed.", min=-1e6, max=1e6, step=0.01),
    ),
    build=None,
    apply=lambda m, ts, c: ts[0].clamp(float(c["min"]), float(c["max"])),
    emit_call=lambda attr, args, c: f"{args[0]}.clamp({float(c['min'])}, {float(c['max'])})",
    trains=False,
)(lambda ins, cfg: same(ins[0]))


# ---------------------------------------------------------------------------
# Matrix multiply
# ---------------------------------------------------------------------------

def _matmul_infer(ins, cfg):
    a, b = ins[0], ins[1]
    if a.rank < 2 or b.rank < 2:
        raise ShapeError(
            f"needs two tensors of rank 2 or more, got {a.shape} and {b.shape}.",
            hint="Matrix multiply works on the last two axes. Use Unsqueeze to add one.",
        )
    if a.dtype != b.dtype:
        raise ShapeError(f"dtypes differ: {a.dtype} and {b.dtype}.")
    inner_a, inner_b = a.shape[-1], b.shape[-2]
    try:
        unify_dim(inner_a, inner_b)
    except ShapeError:
        raise ShapeError(
            f"the inner dimensions do not line up: {a.shape} ends in {inner_a} but {b.shape} "
            f"has {inner_b} in its second-to-last position.",
            hint="For A @ B the last axis of A must equal the second-to-last axis of B. "
                 "A Transpose on one of them is usually what is needed.",
        ) from None
    lead = broadcast([a.shape[:-2], b.shape[:-2]], what="the batch dimensions") if a.rank > 2 or b.rank > 2 else Shape([])
    return out(Shape(list(lead.dims) + [a.shape[-2], b.shape[-1]]), a.dtype)


layer(
    "matmul", "Matrix Multiply", CAT,
    doc="Multiplies the last two axes as matrices, batching over anything in front. This is "
        "the operation a Linear layer is made of, and how attention compares queries to keys.",
    aliases=("matmul", "bmm", "dot product", "@"),
    arity=2,
    input_names=("a", "b"),
    build=None,
    apply=lambda m, ts, c: torch.matmul(ts[0], ts[1]),
    emit_call=lambda attr, args, c: f"torch.matmul({args[0]}, {args[1]})",
    trains=False,
)(_matmul_infer)


# ---------------------------------------------------------------------------
# Einsum — the most general operator here
# ---------------------------------------------------------------------------

_EINSUM_RE = re.compile(r"^\s*([a-zA-Z,\s]+?)\s*->\s*([a-zA-Z\s]*)\s*$")


def _einsum_parse(equation: str, n_inputs: int):
    m = _EINSUM_RE.match(str(equation))
    if not m:
        raise ShapeError(
            f"could not read the einsum equation {equation!r}.",
            hint="Write it with an explicit arrow, e.g. 'bij,bjk->bik'. Ellipsis is not supported here.",
        )
    lhs = [p.strip() for p in m.group(1).split(",")]
    rhs = m.group(2).replace(" ", "")
    if len(lhs) != n_inputs:
        raise ShapeError(
            f"the equation names {len(lhs)} operand(s) but {n_inputs} tensor(s) are connected.",
            hint="Every comma-separated group before the arrow is one input.",
        )
    return lhs, rhs


def _einsum_infer(ins, cfg):
    lhs, rhs = _einsum_parse(cfg["equation"], len(ins))
    sizes: dict[str, Dim] = {}
    for spec, t in zip(lhs, ins):
        if len(spec) != t.rank:
            raise ShapeError(
                f"'{spec}' names {len(spec)} axes but the tensor {t.shape} has {t.rank}.",
                hint="Use one letter per dimension of that input.",
            )
        for letter, d in zip(spec, t.shape.dims):
            if letter in sizes:
                try:
                    sizes[letter] = unify_dim(sizes[letter], d)
                except ShapeError as exc:
                    raise ShapeError(
                        f"the index '{letter}' is used for two different sizes: {exc.message}",
                        hint="Every occurrence of a letter has to refer to the same size.",
                    ) from None
            else:
                sizes[letter] = d
    for letter in rhs:
        if letter not in sizes:
            raise ShapeError(
                f"the output index '{letter}' never appears on the left of the arrow.",
                hint="Output indices must be taken from the inputs.",
            )
    dtypes = {t.dtype for t in ins}
    if len(dtypes) > 1:
        raise ShapeError("all einsum inputs must share a dtype, got " + ", ".join(sorted(dtypes)) + ".")
    return out(Shape([sizes[c] for c in rhs]), ins[0].dtype)


layer(
    "einsum", "Einsum", CAT,
    doc="The general tensor contraction. Name every axis with a letter, say which letters "
        "survive, and any sum-of-products falls out: 'bij,bjk->bik' is a batched matrix "
        "multiply, 'bhqd,bhkd->bhqk' is attention scores.",
    aliases=("einstein summation", "contraction", "tensordot"),
    arity=VARIADIC,
    params=(P("equation", "string", "bij,bjk->bik", "The index equation, with an explicit arrow."),),
    build=None,
    apply=lambda m, ts, c: torch.einsum(str(c["equation"]), *ts),
    emit_call=lambda attr, args, c: f"torch.einsum({str(c['equation'])!r}, {', '.join(args)})",
    trains=False,
)(_einsum_infer)


# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------

_TORCH_DTYPE = {
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
    "int64": torch.int64, "bool": torch.bool,
}


def _cast_infer(ins, cfg):
    target = str(cfg["dtype"])
    if target not in DTYPES:
        raise ShapeError(f"unknown dtype {target!r}.", hint="Choose one of: " + ", ".join(DTYPES))
    return out(ins[0].shape, target)


layer(
    "cast", "Cast", CAT,
    doc="Changes the number type without changing the shape. Mostly needed to turn labels "
        "into indices for an Embedding, or to drop to half precision.",
    aliases=("convert", "dtype", "to float", "to long"),
    params=(P("dtype", "combo", "float32", "The type to convert to.", choices=DTYPES),),
    build=None,
    apply=lambda m, ts, c: ts[0].to(_TORCH_DTYPE[str(c["dtype"])]),
    emit_call=lambda attr, args, c: f"{args[0]}.to(torch.{str(c['dtype'])})",
    trains=False,
)(_cast_infer)


def torch_dtype(name: str):
    """Public lookup used by the compiler and the data loaders."""
    try:
        return _TORCH_DTYPE[name]
    except KeyError:
        raise ShapeError(f"unknown dtype {name!r}.", hint="Choose one of: " + ", ".join(DTYPES)) from None
