"""Reshaping: the plumbing that gets a tensor into the form the next layer wants.

Most "it doesn't work" moments in a first network are a shape mismatch, so these nodes
carry the most detailed error messages in the pack.
"""

from __future__ import annotations

import re

import torch

from ..errors import ShapeError
from ..registry import P, VARIADIC, layer, positive, require_rank
from ..shape import Dim, Shape, normalize_axis, unify_dim
from ._common import out, same

CAT = "shape"


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def _flatten_infer(ins, cfg):
    t = ins[0]
    start = normalize_axis(int(cfg["start_dim"]), t.rank)
    if start >= t.rank - 1:
        return same(t)
    tail = t.shape.dims[start:]
    if all(d.is_concrete for d in tail):
        total = 1
        for d in tail:
            total *= d.size
        merged = Dim(total)
    else:
        names = [str(d) for d in tail if not d.is_concrete]
        raise ShapeError(
            f"cannot flatten {t.shape} from axis {start}, because "
            + ", ".join(f"'{n}'" for n in names) + " has no known size.",
            hint="Flatten multiplies the sizes together, so it needs numbers. Give those "
                 "dimensions concrete sizes, or flatten fewer axes.",
        )
    return out(Shape(list(t.shape.dims[:start]) + [merged]), t.dtype)


layer(
    "flatten", "Flatten", CAT,
    doc="Squashes all the trailing dimensions into one long row, keeping the batch "
        "dimension intact. This is the standard bridge from a stack of image feature maps "
        "into a Linear layer.",
    aliases=("unroll", "vectorise", "vectorize", "to vector"),
    params=(P("start_dim", "int", 1, "First axis to merge. 1 keeps the batch separate.", min=0, max=8),),
    build=None,
    apply=lambda m, ts, c: torch.flatten(ts[0], start_dim=int(c["start_dim"])),
    emit_call=lambda attr, args, c: f"torch.flatten({args[0]}, start_dim={int(c['start_dim'])})",
    trains=False,
)(_flatten_infer)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"^(-1|\d+|[A-Za-z_][A-Za-z_0-9]*)$")


def _reshape_target(text: str, src: Shape) -> Shape:
    parts = [p for p in re.split(r"[,\sx×]+", str(text).strip().strip("[]()")) if p]
    if not parts:
        raise ShapeError("the target shape is empty.",
                         hint="Write the shape you want, e.g. 'B, -1' or 'B, 64, 7, 7'.")
    for p in parts:
        if not _TOKEN.match(p):
            raise ShapeError(f"could not read '{p}' in the target shape '{text}'.",
                             hint="Use numbers, names that already appear in the input, or -1 "
                                  "for one dimension you want worked out for you.")
    if sum(1 for p in parts if p == "-1") > 1:
        raise ShapeError("only one -1 is allowed in a target shape.",
                         hint="Everything except one dimension has to be pinned down.")

    src_names = [str(d) for d in src.dims if d.is_symbolic]
    out_names = [p for p in parts if not re.fullmatch(r"-?\d+", p)]
    for name in out_names:
        if name not in src_names:
            raise ShapeError(
                f"the target shape uses '{name}', which is not a dimension of the input {src}.",
                hint="A name can only be carried through from the input. Input names here are: "
                     + (", ".join(src_names) if src_names else "(none — every input dim has a size)") + ".",
            )
    leftover = list(src_names)
    for name in out_names:
        leftover.remove(name)

    concrete_in = 1
    for d in src.dims:
        if d.is_concrete:
            concrete_in *= d.size
    concrete_out = 1
    for p in parts:
        if re.fullmatch(r"\d+", p):
            concrete_out *= int(p)

    if "-1" in parts:
        if leftover:
            raise ShapeError(
                f"cannot work out the -1: the input {src} still has the unknown dimension(s) "
                + ", ".join(f"'{n}'" for n in leftover) + " that the target drops.",
                hint="Carry those names through to the output, or give them concrete sizes.",
            )
        if concrete_out == 0 or concrete_in % concrete_out:
            raise ShapeError(
                f"cannot reshape {src} to '{text}': {concrete_in} values do not divide evenly "
                f"into the {concrete_out} you asked for.",
                hint=f"The fixed part of the target has to divide {concrete_in} exactly.",
            )
        inferred = concrete_in // concrete_out
        parts = [str(inferred) if p == "-1" else p for p in parts]
    else:
        if leftover or concrete_in != concrete_out:
            detail = (f" and drops the unknown dimension(s) "
                      + ", ".join(f"'{n}'" for n in leftover)) if leftover else ""
            raise ShapeError(
                f"cannot reshape {src} to '{text}': it holds {concrete_in} values per example "
                f"but the target holds {concrete_out}{detail}.",
                hint="Reshape never adds or removes values, it only rearranges them. Use -1 to "
                     "let one dimension be worked out for you.",
            )
    return Shape([int(p) if re.fullmatch(r"\d+", p) else p for p in parts])


def _reshape_infer(ins, cfg):
    return out(_reshape_target(cfg["target"], ins[0].shape), ins[0].dtype)


def _reshape_prepare(shapes, cfg):
    """Turn the target into a runtime plan, now that the input shape is known.

    Each entry is either a concrete size, or ``("axis", k)`` meaning "however big axis k of
    the incoming tensor turns out to be". That second case is what makes a symbolic name
    like ``B`` work at run time without a second -1, which torch would refuse.
    """
    src = shapes[0]
    resolved = _reshape_target(cfg["target"], src)
    src_axis = {str(d): i for i, d in enumerate(src.dims) if d.is_symbolic}
    plan = []
    for d in resolved.dims:
        plan.append(d.size if d.is_concrete else ("axis", src_axis[str(d)]))
    return {"_plan": tuple(plan)}


def _reshape_sizes(tensor, plan):
    return [tensor.shape[p[1]] if isinstance(p, tuple) else int(p) for p in plan]


def _reshape_emit(attr, args, cfg):
    parts = [f"{args[0]}.shape[{p[1]}]" if isinstance(p, tuple) else str(int(p))
             for p in cfg["_plan"]]
    return f"{args[0]}.reshape({', '.join(parts)})"


layer(
    "reshape", "Reshape", CAT,
    doc="Rearranges the same numbers into a different shape. Nothing is added or lost, so "
        "the sizes must multiply to the same total. Write -1 for one dimension and it will "
        "be worked out for you.",
    aliases=("view", "rearrange"),
    params=(P("target", "string", "B, -1", "The shape you want, e.g. 'B, -1' or 'B, 64, 7, 7'."),),
    build=None,
    prepare=_reshape_prepare,
    apply=lambda m, ts, c: ts[0].reshape(*_reshape_sizes(ts[0], c["_plan"])),
    emit_call=_reshape_emit,
    trains=False,
)(_reshape_infer)


# ---------------------------------------------------------------------------
# Permute / transpose
# ---------------------------------------------------------------------------

def _order(cfg, rank: int) -> tuple[int, ...]:
    text = str(cfg["order"]).strip().strip("[]()")
    try:
        vals = tuple(int(p) for p in re.split(r"[,\s]+", text) if p)
    except ValueError:
        raise ShapeError(f"could not read the axis order '{cfg['order']}'.",
                         hint="Write the axis numbers in the order you want, e.g. '0, 2, 3, 1'.") from None
    if sorted(vals) != list(range(rank)):
        raise ShapeError(
            f"the axis order {list(vals)} is not a rearrangement of the {rank} axes of this tensor.",
            hint=f"List each of the numbers 0 to {rank - 1} exactly once.",
        )
    return vals


layer(
    "permute", "Permute", CAT,
    doc="Reorders the dimensions without moving any values between them. Use it to convert "
        "between image layouts, for instance [B, H, W, C] to [B, C, H, W] with '0, 3, 1, 2'.",
    aliases=("transpose axes", "reorder dims", "nhwc", "nchw"),
    params=(P("order", "string", "0, 2, 3, 1", "The axes in the order you want them."),),
    build=None,
    prepare=lambda shapes, cfg: {"_order": _order(cfg, shapes[0].rank)},
    apply=lambda m, ts, c: ts[0].permute(*c["_order"]).contiguous(),
    emit_call=lambda attr, args, c: f"{args[0]}.permute{tuple(c['_order'])}.contiguous()",
    trains=False,
)(lambda ins, cfg: out(Shape([ins[0].shape[i] for i in _order(cfg, ins[0].rank)]), ins[0].dtype))


def _transpose_infer(ins, cfg):
    t = ins[0]
    a = normalize_axis(int(cfg["dim0"]), t.rank)
    b = normalize_axis(int(cfg["dim1"]), t.rank)
    dims = list(t.shape.dims)
    dims[a], dims[b] = dims[b], dims[a]
    return out(Shape(dims), t.dtype)


layer(
    "transpose", "Transpose", CAT,
    doc="Swaps exactly two dimensions. The common use is turning [batch, time, features] "
        "into [batch, features, time] so a Conv 1D can read it.",
    aliases=("swap axes",),
    params=(
        P("dim0", "int", 1, "First axis to swap.", min=-8, max=8),
        P("dim1", "int", 2, "Second axis to swap.", min=-8, max=8),
    ),
    build=None,
    apply=lambda m, ts, c: ts[0].transpose(int(c["dim0"]), int(c["dim1"])).contiguous(),
    emit_call=lambda attr, args, c: f"{args[0]}.transpose({int(c['dim0'])}, {int(c['dim1'])}).contiguous()",
    trains=False,
)(_transpose_infer)


# ---------------------------------------------------------------------------
# Squeeze / unsqueeze
# ---------------------------------------------------------------------------

def _unsqueeze_infer(ins, cfg):
    t = ins[0]
    axis = normalize_axis(int(cfg["dim"]), t.rank, allow_end=True)
    return out(t.shape.insert(axis, 1), t.dtype)


layer(
    "unsqueeze", "Unsqueeze", CAT,
    doc="Inserts a dimension of size 1. Use it to give a greyscale image its missing channel "
        "axis, or to line two tensors up for broadcasting.",
    aliases=("expand dims", "add axis"),
    params=(P("dim", "int", 1, "Where to insert the new axis.", min=-8, max=8),),
    build=None,
    apply=lambda m, ts, c: ts[0].unsqueeze(int(c["dim"])),
    emit_call=lambda attr, args, c: f"{args[0]}.unsqueeze({int(c['dim'])})",
    trains=False,
)(_unsqueeze_infer)


def _squeeze_infer(ins, cfg):
    t = ins[0]
    axis = normalize_axis(int(cfg["dim"]), t.rank)
    d = t.shape[axis]
    if d.is_concrete and d.size != 1:
        raise ShapeError(
            f"axis {axis} of {t.shape} has size {d}, and only a size-1 axis can be squeezed out.",
            hint="Pick an axis whose size is 1, or use Reduce if you want to collapse a real dimension.",
        )
    return out(t.shape.drop(axis), t.dtype)


layer(
    "squeeze", "Squeeze", CAT,
    doc="Removes a dimension of size 1. The opposite of Unsqueeze.",
    aliases=("drop axis", "remove dim"),
    params=(P("dim", "int", 1, "Which axis to remove.", min=-8, max=8),),
    build=None,
    apply=lambda m, ts, c: ts[0].squeeze(int(c["dim"])),
    emit_call=lambda attr, args, c: f"{args[0]}.squeeze({int(c['dim'])})",
    trains=False,
)(_squeeze_infer)


# ---------------------------------------------------------------------------
# Concat / stack
# ---------------------------------------------------------------------------

def _concat_infer(ins, cfg):
    ranks = {t.rank for t in ins}
    if len(ranks) > 1:
        raise ShapeError(
            "every input must have the same number of dimensions, but got "
            + " and ".join(str(t.shape) for t in ins) + ".",
            hint="Reshape or Unsqueeze the odd one out so all the ranks match.",
        )
    rank = ins[0].rank
    axis = normalize_axis(int(cfg["dim"]), rank)
    dtypes = {t.dtype for t in ins}
    if len(dtypes) > 1:
        raise ShapeError("every input must have the same dtype, got " + ", ".join(sorted(dtypes)) + ".")
    dims = []
    for i in range(rank):
        if i == axis:
            continue
        acc = ins[0].shape[i]
        for other in ins[1:]:
            try:
                acc = unify_dim(acc, other.shape[i], where=f" (axis {i})")
            except ShapeError as exc:
                raise ShapeError(
                    f"the inputs disagree away from the join axis: {exc.message}",
                    hint="Concat only lets axis "
                         f"{axis} differ. Every other axis has to match exactly. Shapes are "
                         + " and ".join(str(t.shape) for t in ins) + ".",
                ) from None
        dims.append(acc)
    joined = [t.shape[axis] for t in ins]
    if all(d.is_concrete for d in joined):
        total = Dim(sum(d.size for d in joined))
    else:
        raise ShapeError(
            f"cannot add up the sizes along axis {axis}, because "
            + ", ".join(f"'{d}'" for d in joined if not d.is_concrete) + " has no known size.",
            hint="Give the joined axis a concrete size, or join along a different axis.",
        )
    result = list(dims)
    result.insert(axis, total)
    return out(Shape(result), ins[0].dtype)


layer(
    "concat", "Concat", CAT,
    doc="Glues tensors together along one axis, so the features of each are kept side by "
        "side. This is how a U-Net reunites its encoder and decoder paths.",
    aliases=("concatenate", "join", "merge", "cat", "skip connection"),
    arity=VARIADIC,
    params=(P("dim", "int", 1, "The axis to join along.", min=-8, max=8),),
    build=None,
    apply=lambda m, ts, c: torch.cat(ts, dim=int(c["dim"])),
    emit_call=lambda attr, args, c: f"torch.cat([{', '.join(args)}], dim={int(c['dim'])})",
    trains=False,
)(_concat_infer)


def _stack_infer(ins, cfg):
    base = ins[0].shape
    for other in ins[1:]:
        if other.rank != base.rank:
            raise ShapeError("Stack needs every input to have the same shape, got "
                             + " and ".join(str(t.shape) for t in ins) + ".")
        for i in range(base.rank):
            unify_dim(base[i], other.shape[i], where=f" (axis {i})")
    axis = normalize_axis(int(cfg["dim"]), base.rank, allow_end=True)
    return out(base.insert(axis, len(ins)), ins[0].dtype)


layer(
    "stack", "Stack", CAT,
    doc="Puts tensors of identical shape onto a brand new axis, unlike Concat which widens "
        "an existing one.",
    aliases=("pile", "new axis"),
    arity=VARIADIC,
    params=(P("dim", "int", 1, "Where the new axis goes.", min=-8, max=8),),
    build=None,
    apply=lambda m, ts, c: torch.stack(ts, dim=int(c["dim"])),
    emit_call=lambda attr, args, c: f"torch.stack([{', '.join(args)}], dim={int(c['dim'])})",
    trains=False,
)(_stack_infer)


# ---------------------------------------------------------------------------
# Slice / reduce
# ---------------------------------------------------------------------------

def _slice_bounds(cfg, d: Dim):
    start, end = int(cfg["start"]), int(cfg["end"])
    if not d.is_concrete:
        return start, end, None
    n = d.size
    s = start + n if start < 0 else start
    e = n if end == 0 or end > n else (end + n if end < 0 else end)
    if not 0 <= s < n:
        raise ShapeError(f"start {start} is outside a dimension of size {n}.",
                         hint=f"Use a value from {-n} to {n - 1}.")
    if e <= s:
        raise ShapeError(f"the slice from {start} to {end} is empty on a dimension of size {n}.",
                         hint="'end' must be past 'start'. Leave end at 0 to run to the end.")
    return s, e, e - s


def _slice_infer(ins, cfg):
    t = ins[0]
    axis = normalize_axis(int(cfg["dim"]), t.rank)
    _, _, length = _slice_bounds(cfg, t.shape[axis])
    if length is None:
        raise ShapeError(f"cannot slice the symbolic dimension '{t.shape[axis]}'.",
                         hint="Slice an axis whose size is known.")
    return out(t.shape.replace(axis, length), t.dtype)


def _slice_prepare(shapes, cfg):
    axis = normalize_axis(int(cfg["dim"]), shapes[0].rank)
    start, _, length = _slice_bounds(cfg, shapes[0][axis])
    return {"_axis": axis, "_start": start, "_length": length}


def _slice_emit(attr, args, c):
    parts = [":"] * (c["_axis"] + 1)
    parts[c["_axis"]] = f"{c['_start'] or ''}:{c['_start'] + c['_length']}"
    return f"{args[0]}[{', '.join(parts)}]"


layer(
    "slice", "Slice", CAT,
    doc="Takes a contiguous piece of one axis. Common uses are picking the last timestep of "
        "a sequence, or splitting a tensor into halves.",
    aliases=("crop", "take", "narrow", "index"),
    params=(
        P("dim", "int", 1, "The axis to cut along.", min=-8, max=8),
        P("start", "int", 0, "First index to keep. Negative counts from the end.", min=-1 << 20, max=1 << 20),
        P("end", "int", 0, "One past the last index. 0 means run to the end.", min=-1 << 20, max=1 << 20),
    ),
    build=None,
    prepare=_slice_prepare,
    apply=lambda m, ts, c: ts[0].narrow(c["_axis"], c["_start"], c["_length"]),
    emit_call=_slice_emit,
    trains=False,
)(_slice_infer)


_REDUCE = {
    "mean": (torch.mean, "mean"), "sum": (torch.sum, "sum"),
    "max": (torch.amax, "amax"), "min": (torch.amin, "amin"),
    "prod": (torch.prod, "prod"),
}


def _reduce_infer(ins, cfg):
    t = ins[0]
    axis = normalize_axis(int(cfg["dim"]), t.rank)
    if cfg["mode"] not in _REDUCE:
        raise ShapeError(f"unknown reduction {cfg['mode']!r}.",
                         hint="Choose one of: " + ", ".join(_REDUCE))
    return out(t.shape.replace(axis, 1) if bool(cfg["keepdim"]) else t.shape.drop(axis), t.dtype)


layer(
    "reduce", "Reduce", CAT,
    doc="Collapses one axis into a single value using an average, a sum or an extreme. "
        "Averaging over the time axis is the simplest way to pool a sequence into one vector.",
    aliases=("mean", "sum", "pool", "aggregate"),
    params=(
        P("mode", "combo", "mean", "How to combine the values.", choices=tuple(_REDUCE)),
        P("dim", "int", 1, "The axis to collapse.", min=-8, max=8),
        P("keepdim", "bool", False, "Leave a size-1 axis behind instead of removing it."),
    ),
    build=None,
    apply=lambda m, ts, c: _REDUCE[c["mode"]][0](ts[0], dim=int(c["dim"]), keepdim=bool(c["keepdim"])),
    emit_call=lambda attr, args, c: "torch.{}({}, dim={}, keepdim={})".format(
        _REDUCE[c["mode"]][1], args[0], int(c["dim"]), bool(c["keepdim"])),
    trains=False,
)(_reduce_infer)
