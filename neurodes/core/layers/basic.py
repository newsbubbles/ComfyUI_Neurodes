"""Dense layers, embeddings, dropout, normalisation, and the plain reshaping ones."""

from __future__ import annotations

import torch
from torch import nn

from ..errors import ShapeError
from ..registry import (P, VARIADIC, layer, positive, require_float, require_rank)
from ..shape import Dim, Shape
from ..trace import SymTensor
from ._common import fmt_num, kwargs_src, out, same

CAT = "layers/basic"
CAT_NORM = "layers/norm"


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def _linear_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Linear")
    require_rank(t, 1, "(at least a batch and a feature dimension)", at_least=True)
    units = positive(cfg["units"], "units")
    last = t.shape[-1]
    if not last.is_concrete:
        raise ShapeError(
            f"the last dimension of {t.shape} is the symbolic size '{last}', so Linear cannot "
            "know how many weights to make.",
            hint=f"Give '{last}' a concrete size in the Input node, or move this layer after a "
                 "Flatten so the feature dimension is known.",
        )
    return out(t.shape.replace(-1, units), t.dtype)


def _linear_build(shapes, cfg):
    return nn.Linear(shapes[0][-1].size, positive(cfg["units"], "units"), bias=bool(cfg["bias"]))


layer(
    "linear", "Linear", CAT,
    doc="A fully connected layer. Every value in comes from every value out, through a "
        "learned weight. Only the last dimension changes; how many inputs it takes is read "
        "off the incoming shape, so you only choose the width.",
    aliases=("dense", "fully connected", "fc", "mlp", "matmul layer"),
    params=(
        P("units", "int", 128, "How wide the output is.", min=1, max=1 << 20),
        P("bias", "bool", True, "Add a learned offset per output."),
    ),
    build=_linear_build,
    emit_init=lambda s, c: f"nn.Linear({s[0][-1].size}, {int(c['units'])}, {kwargs_src(bias=bool(c['bias']))})",
)(_linear_infer)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embedding_infer(ins, cfg):
    t = ins[0]
    if t.dtype != "int64":
        raise ShapeError(
            f"Embedding takes token indices (int64) but got {t.dtype}.",
            hint="Set the Input node's dtype to int64, or put a Cast to int64 before this layer.",
        )
    size = positive(cfg["embedding_dim"], "embedding_dim")
    return out(Shape(list(t.shape.dims) + [Dim(size)]), "float32")


layer(
    "embedding", "Embedding", CAT,
    doc="A lookup table. Turns whole-number ids (words, tokens, categories) into learned "
        "vectors, adding one dimension of that width to the end of the shape.",
    aliases=("lookup table", "token embedding", "word vectors"),
    params=(
        P("num_embeddings", "int", 1000, "How many distinct ids exist.", min=1, max=1 << 24),
        P("embedding_dim", "int", 64, "The width of each learned vector.", min=1, max=1 << 16),
        P("padding_idx", "int", -1, "Id whose vector stays zero. -1 disables.", min=-1, max=1 << 24, advanced=True),
    ),
    build=lambda s, c: nn.Embedding(
        positive(c["num_embeddings"], "num_embeddings"),
        positive(c["embedding_dim"], "embedding_dim"),
        padding_idx=None if int(c["padding_idx"]) < 0 else int(c["padding_idx"]),
    ),
    emit_init=lambda s, c: "nn.Embedding({}, {}{})".format(
        int(c["num_embeddings"]), int(c["embedding_dim"]),
        "" if int(c["padding_idx"]) < 0 else f", padding_idx={int(c['padding_idx'])}",
    ),
)(_embedding_infer)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

layer(
    "dropout", "Dropout", CAT,
    doc="During training only, randomly zeroes some values. Stops the network leaning on "
        "any single path. Does nothing at evaluation time.",
    aliases=("regularisation", "regularization"),
    params=(P("p", "float", 0.1, "Fraction of values dropped.", min=0.0, max=0.99, step=0.01),),
    build=lambda s, c: nn.Dropout(float(c["p"])),
    emit_init=lambda s, c: f"nn.Dropout({float(c['p'])})",
    trains=False,
)(lambda ins, cfg: same(ins[0]))


layer(
    "identity", "Identity", CAT,
    doc="Passes the tensor through unchanged. Useful as a placeholder while you build.",
    aliases=("passthrough", "noop"),
    build=None,
    apply=lambda m, ts, c: ts[0],
    emit_call=lambda attr, args, c: args[0],
    trains=False,
)(lambda ins, cfg: same(ins[0]))


# ---------------------------------------------------------------------------
# Normalisation. One node each, but the rank picks the torch class, so nobody has to
# know the difference between BatchNorm1d and BatchNorm2d to use batch norm.
# ---------------------------------------------------------------------------

_BN = {2: nn.BatchNorm1d, 3: nn.BatchNorm1d, 4: nn.BatchNorm2d, 5: nn.BatchNorm3d}


def _batchnorm_infer(ins, cfg):
    t = ins[0]
    require_float(t, "BatchNorm")
    if t.rank not in _BN:
        raise ShapeError(
            f"BatchNorm works on rank 2 to 5 tensors, but got {t.shape} (rank {t.rank}).",
            hint="Batch norm normalises over the channel dimension, which is axis 1.",
        )
    if not t.shape[1].is_concrete:
        raise ShapeError(
            f"BatchNorm needs a concrete channel count, but axis 1 of {t.shape} is '{t.shape[1]}'.",
            hint="Give the channel dimension a number in the Input node.",
        )
    return same(t)


def _bn_class_name(rank: int) -> str:
    return {2: "BatchNorm1d", 3: "BatchNorm1d", 4: "BatchNorm2d", 5: "BatchNorm3d"}[rank]


layer(
    "batchnorm", "Batch Norm", CAT_NORM,
    doc="Rescales each channel using the mean and variance of the batch, which keeps the "
        "numbers in a sane range and makes training much faster. Picks the 1d/2d/3d "
        "variant from the shape you give it.",
    aliases=("batchnorm1d", "batchnorm2d", "bn", "normalization"),
    params=(
        P("momentum", "float", 0.1, "How fast the running statistics follow the batch.",
          min=0.0, max=1.0, step=0.01, advanced=True),
        P("affine", "bool", True, "Learn a scale and shift per channel.", advanced=True),
    ),
    build=lambda s, c: _BN[s[0].rank](s[0][1].size, momentum=float(c["momentum"]), affine=bool(c["affine"])),
    emit_init=lambda s, c: "nn.{}({}, {})".format(
        _bn_class_name(s[0].rank), s[0][1].size,
        kwargs_src(momentum=float(c["momentum"]), affine=bool(c["affine"])),
    ),
)(_batchnorm_infer)


def _layernorm_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Layer Norm")
    n = positive(cfg["last_dims"], "last_dims")
    if n > t.rank:
        raise ShapeError(
            f"Layer Norm was told to normalise the last {n} dimensions, but {t.shape} only has {t.rank}.",
            hint="Lower 'last_dims', or send in a tensor with more dimensions.",
        )
    for d in t.shape.dims[-n:]:
        if not d.is_concrete:
            raise ShapeError(
                f"Layer Norm needs concrete sizes for the last {n} dimension(s) of {t.shape}, "
                f"but found '{d}'.",
                hint="Normalising over a dimension requires knowing how big it is.",
            )
    return same(t)


def _ln_shape(s: Shape, n: int) -> list[int]:
    return [d.size for d in s.dims[-n:]]


layer(
    "layernorm", "Layer Norm", CAT_NORM,
    doc="Rescales each example on its own, using the mean and variance across its last "
        "dimensions. Unlike batch norm it does not care about the other examples, which "
        "is why transformers use it.",
    aliases=("ln", "layer normalization"),
    params=(
        P("last_dims", "int", 1, "How many trailing dimensions to normalise over.", min=1, max=4),
        P("eps", "float", 1e-5, "Guard against dividing by zero.", min=1e-12, max=1e-1, step=1e-6, advanced=True),
    ),
    build=lambda s, c: nn.LayerNorm(_ln_shape(s[0], positive(c["last_dims"], "last_dims")), eps=float(c["eps"])),
    emit_init=lambda s, c: "nn.LayerNorm({}, eps={})".format(
        fmt_num(tuple(_ln_shape(s[0], int(c["last_dims"])))), float(c["eps"])),
)(_layernorm_infer)


def _rmsnorm_infer(ins, cfg):
    t = ins[0]
    require_float(t, "RMS Norm")
    if not t.shape[-1].is_concrete:
        raise ShapeError(
            f"RMS Norm needs a concrete last dimension, but {t.shape} ends in '{t.shape[-1]}'.",
            hint="Give the feature dimension a number.",
        )
    return same(t)


layer(
    "rmsnorm", "RMS Norm", CAT_NORM,
    doc="Layer norm without the mean subtraction: divides by the root-mean-square of the "
        "last dimension. Cheaper, and what most recent language models use.",
    aliases=("rms", "root mean square norm"),
    params=(P("eps", "float", 1e-6, "Guard against dividing by zero.", min=1e-12, max=1e-1, step=1e-8, advanced=True),),
    build=lambda s, c: nn.RMSNorm(s[0][-1].size, eps=float(c["eps"])),
    emit_init=lambda s, c: f"nn.RMSNorm({s[0][-1].size}, eps={float(c['eps'])})",
)(_rmsnorm_infer)


def _groupnorm_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Group Norm")
    require_rank(t, 3, "(batch, channels, ...)", at_least=True)
    if not t.shape[1].is_concrete:
        raise ShapeError(f"Group Norm needs a concrete channel count, got '{t.shape[1]}'.")
    channels, groups = t.shape[1].size, positive(cfg["groups"], "groups")
    if channels % groups:
        raise ShapeError(
            f"Group Norm cannot split {channels} channels into {groups} equal groups.",
            hint="Pick a group count that divides the channel count exactly, e.g. "
                 + ", ".join(str(g) for g in _divisors(channels)[:6]) + ".",
        )
    return same(t)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


layer(
    "groupnorm", "Group Norm", CAT_NORM,
    doc="Splits the channels into groups and normalises inside each one. Behaves the same "
        "whatever the batch size, which batch norm does not.",
    aliases=("gn",),
    params=(P("groups", "int", 8, "How many channel groups.", min=1, max=1024),),
    build=lambda s, c: nn.GroupNorm(positive(c["groups"], "groups"), s[0][1].size),
    emit_init=lambda s, c: f"nn.GroupNorm({int(c['groups'])}, {s[0][1].size})",
)(_groupnorm_infer)
