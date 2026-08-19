"""Activation functions.

These are the layers that bend the line. Without one of them between two Linear layers,
the whole stack collapses back into a single Linear — which is the single most useful fact
a beginner can learn about architecture, so every node here says so.

Stateless activations emit as ``F.relu(x)`` rather than a module in ``__init__``, because
that is what hand-written PyTorch actually looks like in a forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import ShapeError
from ..registry import P, layer, require_float
from ..shape import normalize_axis
from ._common import out, same

CAT = "activations"

_WHY = ("Between two Linear layers this is what stops the pair collapsing into one "
        "Linear. Without it, depth buys you nothing.")


def _simple(key: str, display: str, fn, src: str, doc: str, aliases=(), params=(), extra=None):
    """Register a stateless, shape-preserving activation."""

    def infer(ins, cfg):
        require_float(ins[0], display)
        return same(ins[0])

    layer(
        key, display, CAT,
        doc=doc + " " + _WHY,
        aliases=aliases,
        params=params,
        build=None,
        apply=lambda m, ts, c, _fn=fn: _fn(ts[0], c),
        emit_call=lambda attr, args, c, _s=src: _s.format(x=args[0], **{k: _fmt(v) for k, v in c.items()}),
        trains=False,
    )(infer)


def _fmt(v):
    return "True" if v is True else "False" if v is False else v


_simple("relu", "ReLU", lambda x, c: F.relu(x), "F.relu({x})",
        "Keeps positive values, sets negative ones to zero. The default choice, and the "
        "fastest.", aliases=("rectifier",))

_simple("leaky_relu", "Leaky ReLU", lambda x, c: F.leaky_relu(x, float(c["negative_slope"])),
        "F.leaky_relu({x}, {negative_slope})",
        "Like ReLU, but negative values are scaled down instead of erased, so neurons that "
        "go negative can still recover.",
        params=(P("negative_slope", "float", 0.01, "Slope for negative inputs.", min=0.0, max=1.0, step=0.001),))

_simple("gelu", "GELU", lambda x, c: F.gelu(x), "F.gelu({x})",
        "A smooth ReLU. Standard inside transformers.", aliases=("gaussian error linear unit",))

_simple("silu", "SiLU", lambda x, c: F.silu(x), "F.silu({x})",
        "x times sigmoid(x). Smooth, and the usual choice in modern vision and language "
        "models.", aliases=("swish",))

_simple("mish", "Mish", lambda x, c: F.mish(x), "F.mish({x})",
        "A smooth activation that keeps a small negative tail.")

_simple("elu", "ELU", lambda x, c: F.elu(x, float(c["alpha"])), "F.elu({x}, {alpha})",
        "Smoothly saturates to a negative value instead of clipping to zero.",
        params=(P("alpha", "float", 1.0, "How far negative it saturates.", min=0.0, max=10.0, step=0.1),))

_simple("selu", "SELU", lambda x, c: F.selu(x), "F.selu({x})",
        "A self-normalising activation: with the right initialisation it keeps activations "
        "near zero mean and unit variance on its own.")

_simple("tanh", "Tanh", lambda x, c: torch.tanh(x), "torch.tanh({x})",
        "Squashes everything into -1 to 1. Common in small networks and RNNs.")

_simple("sigmoid", "Sigmoid", lambda x, c: torch.sigmoid(x), "torch.sigmoid({x})",
        "Squashes everything into 0 to 1, so the output reads as a probability. Use it on "
        "the final layer of a yes/no network.", aliases=("logistic",))

_simple("softplus", "Softplus", lambda x, c: F.softplus(x), "F.softplus({x})",
        "A smooth ReLU that is always positive. Handy when an output must not be negative.")

_simple("hardswish", "Hard Swish", lambda x, c: F.hardswish(x), "F.hardswish({x})",
        "A cheap piecewise approximation of SiLU, built for phones.")

_simple("relu6", "ReLU6", lambda x, c: F.relu6(x), "F.relu6({x})",
        "ReLU with a ceiling at 6, which keeps activations in a fixed range.")


# ---------------------------------------------------------------------------
# The ones that need an axis
# ---------------------------------------------------------------------------

def _axis_infer(display: str):
    def infer(ins, cfg):
        t = ins[0]
        require_float(t, display)
        normalize_axis(int(cfg["dim"]), t.rank)
        return same(t)
    return infer


layer(
    "softmax", "Softmax", CAT,
    doc="Turns a row of numbers into a set of probabilities that add up to 1. Put it on "
        "the last layer of a classifier — but note that Cross Entropy loss applies it for "
        "you, so adding both is a common and quiet mistake.",
    aliases=("probabilities", "classifier head"),
    params=(P("dim", "int", -1, "Which axis the probabilities add up across.", min=-8, max=8),),
    build=None,
    apply=lambda m, ts, c: F.softmax(ts[0], dim=int(c["dim"])),
    emit_call=lambda attr, args, c: f"F.softmax({args[0]}, dim={int(c['dim'])})",
    trains=False,
)(_axis_infer("Softmax"))

layer(
    "log_softmax", "Log Softmax", CAT,
    doc="Softmax followed by a logarithm, done in a numerically safe way. Pairs with NLL loss.",
    params=(P("dim", "int", -1, "Which axis to normalise across.", min=-8, max=8),),
    build=None,
    apply=lambda m, ts, c: F.log_softmax(ts[0], dim=int(c["dim"])),
    emit_call=lambda attr, args, c: f"F.log_softmax({args[0]}, dim={int(c['dim'])})",
    trains=False,
)(_axis_infer("Log Softmax"))


# ---------------------------------------------------------------------------
# The one with weights
# ---------------------------------------------------------------------------

def _prelu_infer(ins, cfg):
    t = ins[0]
    require_float(t, "PReLU")
    if bool(cfg["per_channel"]):
        if t.rank < 2:
            raise ShapeError("PReLU per-channel needs at least a batch and a channel dimension.",
                             hint="Turn off 'per_channel', or send a rank-2 or larger tensor.")
        if not t.shape[1].is_concrete:
            raise ShapeError(f"PReLU per-channel needs a concrete channel count, got '{t.shape[1]}'.")
    return same(t)


layer(
    "prelu", "PReLU", CAT,
    doc="Leaky ReLU where the network learns the negative slope for itself. "
        "This activation has weights, unlike the others.",
    aliases=("parametric relu",),
    params=(P("per_channel", "bool", False, "One learned slope per channel instead of one overall."),),
    build=lambda s, c: nn.PReLU(num_parameters=s[0][1].size if bool(c["per_channel"]) else 1),
    emit_init=lambda s, c: f"nn.PReLU(num_parameters={s[0][1].size if bool(c['per_channel']) else 1})",
)(_prelu_infer)


#: Activation names usable as a sub-choice inside composite blocks.
BLOCK_ACTIVATIONS = ("relu", "gelu", "silu", "tanh", "leaky_relu", "elu", "mish", "none")


def apply_named(name: str, x):
    """Run an activation by registry key. Used by the composite blocks."""
    table = {
        "relu": F.relu, "gelu": F.gelu, "silu": F.silu, "tanh": torch.tanh,
        "leaky_relu": F.leaky_relu, "elu": F.elu, "mish": F.mish,
        "sigmoid": torch.sigmoid, "none": lambda t: t,
    }
    if name not in table:
        raise ShapeError(f"Unknown activation {name!r}.",
                         hint="Choose one of: " + ", ".join(table))
    return table[name](x)


def named_src(name: str, expr: str) -> str:
    """Source text for an activation applied to ``expr``."""
    if name == "none":
        return expr
    if name in ("tanh", "sigmoid"):
        return f"torch.{name}({expr})"
    return f"F.{name}({expr})"
