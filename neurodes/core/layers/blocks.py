"""Composite blocks — the unit people actually think in.

Nobody draws twelve identical transformer layers. They draw one and write "x12". These
nodes are that, and every one of them can be rebuilt from the primitives in the other
modules if you want to see inside. The exported source shows exactly what they contain.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn

from ..errors import ShapeError
from ..registry import P, layer, positive, require_float, require_rank
from ..shape import Dim, Shape, conv_out
from . import modules
from ._common import kwargs_src, out, same

CAT = "blocks"

_ACTS = ("relu", "gelu", "silu", "tanh", "leaky_relu", "elu", "mish", "none")
_ACTS_SRC = "ACTS = {\n" + "".join(
    f"    {k!r}: {v},\n" for k, v in [
        ("relu", "F.relu"), ("gelu", "F.gelu"), ("silu", "F.silu"), ("tanh", "torch.tanh"),
        ("leaky_relu", "F.leaky_relu"), ("elu", "F.elu"), ("mish", "F.mish"),
        ("sigmoid", "torch.sigmoid"), ("none", "lambda t: t"),
    ]) + "}\n"


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def _mlp_infer(ins, cfg):
    t = ins[0]
    require_float(t, "MLP Block")
    require_rank(t, 2, "(batch, features)", at_least=True)
    if not t.shape[-1].is_concrete:
        raise ShapeError(
            f"needs a concrete feature width, but {t.shape} ends in '{t.shape[-1]}'.",
            hint="Flatten first, or give the last dimension a number.",
        )
    positive(cfg["hidden"], "hidden")
    positive(cfg["depth"], "depth")
    return out(t.shape.replace(-1, positive(cfg["out_features"], "out_features")), t.dtype)


layer(
    "mlp_block", "MLP Block", CAT,
    doc="A stack of Linear layers with an activation between each pair. The plainest "
        "possible network, and still the right answer for a lot of tabular problems. "
        "'depth' counts the Linear layers, so depth 2 is one hidden layer.",
    aliases=("feedforward", "dense block", "multilayer perceptron", "ffn", "head"),
    params=(
        P("hidden", "int", 128, "Width of the hidden layers.", min=1, max=1 << 18),
        P("out_features", "int", 10, "Width of the output.", min=1, max=1 << 18),
        P("depth", "int", 2, "How many Linear layers in total.", min=1, max=32),
        P("activation", "combo", "relu", "Which activation to use between layers.", choices=_ACTS),
        P("dropout", "float", 0.0, "Dropout between layers.", min=0.0, max=0.9, step=0.05),
        P("norm", "bool", False, "Layer-normalise between layers.", advanced=True),
    ),
    build=lambda s, c: modules.MLPBlock(
        s[0][-1].size, positive(c["hidden"], "hidden"), positive(c["out_features"], "out_features"),
        depth=positive(c["depth"], "depth"), activation=str(c["activation"]),
        dropout=float(c["dropout"]), norm=bool(c["norm"])),
    emit_init=lambda s, c: "MLPBlock({}, {}, {}, {})".format(
        s[0][-1].size, int(c["hidden"]), int(c["out_features"]),
        kwargs_src(depth=int(c["depth"]), activation=str(c["activation"]),
                   dropout=float(c["dropout"]), norm=bool(c["norm"]))),
    helpers=(_ACTS_SRC, inspect.getsource(modules.MLPBlock)),
)(_mlp_infer)


# ---------------------------------------------------------------------------
# Conv block
# ---------------------------------------------------------------------------

def _conv_block_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Conv Block")
    if t.rank != 4:
        raise ShapeError(
            f"expects [batch, channels, height, width], but got {t.shape} (rank {t.rank}).",
            hint="This block is for images. Use Unsqueeze to add a channel axis to a "
                 "greyscale image, or MLP Block for flat data.",
        )
    if not t.shape[1].is_concrete:
        raise ShapeError(f"needs a concrete channel count, got '{t.shape[1]}'.")
    kernel = positive(cfg["kernel_size"], "kernel_size")
    stride = positive(cfg["stride"], "stride")
    out_ch = positive(cfg["out_channels"], "out_channels")
    groups = positive(cfg["groups"], "groups")
    if t.shape[1].size % groups or out_ch % groups:
        raise ShapeError(
            f"groups={groups} does not divide both {t.shape[1].size} input and {out_ch} output channels.",
            hint="Use groups=1, or a value that divides both.")
    dims = [t.shape[0], Dim(out_ch)]
    for i, name in ((2, "height"), (3, "width")):
        d = conv_out(t.shape[i], kernel, stride, (kernel - 1) // 2, 1, axis_name=name)
        if bool(cfg["pool"]):
            d = conv_out(d, 2, 2, 0, 1, axis_name=name)
        dims.append(d)
    return out(Shape(dims), t.dtype)


layer(
    "conv_block", "Conv Block", CAT,
    doc="Convolution, then normalisation, then activation — the three-step unit that almost "
        "every vision network is built from. Turn on 'pool' to also halve the picture, which "
        "is how a network trades resolution for depth of meaning.",
    aliases=("cnn block", "vgg block", "vision"),
    params=(
        P("out_channels", "int", 32, "How many feature maps out.", min=1, max=1 << 16),
        P("kernel_size", "int", 3, "Window size. Padding keeps the size the same.", min=1, max=15, step=2),
        P("stride", "int", 1, "2 halves the picture.", min=1, max=8),
        P("norm", "combo", "batch", "Which normalisation to use.", choices=("batch", "group", "none")),
        P("activation", "combo", "relu", "Which activation to use.", choices=_ACTS),
        P("pool", "bool", False, "Add a 2x2 max pool at the end, halving height and width."),
        P("groups", "int", 1, "Split the channels into groups.", min=1, max=1 << 14, advanced=True),
    ),
    build=lambda s, c: modules.ConvBlock(
        s[0][1].size, positive(c["out_channels"], "out_channels"),
        kernel_size=int(c["kernel_size"]), stride=int(c["stride"]), norm=str(c["norm"]),
        activation=str(c["activation"]), pool=bool(c["pool"]), groups=int(c["groups"])),
    emit_init=lambda s, c: "ConvBlock({}, {}, {})".format(
        s[0][1].size, int(c["out_channels"]),
        kwargs_src(kernel_size=int(c["kernel_size"]), stride=int(c["stride"]), norm=str(c["norm"]),
                   activation=str(c["activation"]), pool=bool(c["pool"]), groups=int(c["groups"]))),
    helpers=(_ACTS_SRC, inspect.getsource(modules.ConvBlock)),
)(_conv_block_infer)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

def _residual_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Residual Block")
    if t.rank != 4:
        raise ShapeError(
            f"expects [batch, channels, height, width], but got {t.shape} (rank {t.rank}).",
            hint="For flat data, build a residual by hand: MLP Block into an Add with the input.",
        )
    if not t.shape[1].is_concrete:
        raise ShapeError(f"needs a concrete channel count, got '{t.shape[1]}'.")
    stride = positive(cfg["stride"], "stride")
    out_ch = int(cfg["out_channels"]) or t.shape[1].size
    positive(out_ch, "out_channels")
    dims = [t.shape[0], Dim(out_ch)]
    for i, name in ((2, "height"), (3, "width")):
        dims.append(conv_out(t.shape[i], 3, stride, 1, 1, axis_name=name))
    return out(Shape(dims), t.dtype)


layer(
    "residual_block", "Residual Block", CAT,
    doc="Two convolutions with the input added back onto the result. The addition is the "
        "whole trick: it gives the gradient a clear path backwards, which is why networks "
        "went from about 20 layers deep to hundreds. When the shape changes, the shortcut "
        "becomes a 1x1 convolution so the two sides can still be added.",
    aliases=("resnet", "skip block", "shortcut block"),
    params=(
        P("out_channels", "int", 0, "Channels out. 0 keeps the input's channel count.", min=0, max=1 << 16),
        P("stride", "int", 1, "2 halves the picture and switches the shortcut to a 1x1 convolution.", min=1, max=4),
        P("activation", "combo", "relu", "Which activation to use.", choices=_ACTS),
        P("norm", "combo", "batch", "Which normalisation to use.", choices=("batch", "group", "none")),
    ),
    build=lambda s, c: modules.ResidualBlock(
        s[0][1].size, out_channels=int(c["out_channels"]), stride=int(c["stride"]),
        activation=str(c["activation"]), norm=str(c["norm"])),
    emit_init=lambda s, c: "ResidualBlock({}, {})".format(
        s[0][1].size,
        kwargs_src(out_channels=int(c["out_channels"]), stride=int(c["stride"]),
                   activation=str(c["activation"]), norm=str(c["norm"]))),
    helpers=(_ACTS_SRC, inspect.getsource(modules.ResidualBlock)),
)(_residual_infer)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

def _transformer_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Transformer Block")
    if t.rank != 3:
        raise ShapeError(
            f"expects [batch, time, features], but got {t.shape} (rank {t.rank}).",
            hint="A transformer reads a sequence. An Embedding turns [batch, time] token ids "
                 "into this shape.",
        )
    if not t.shape[2].is_concrete:
        raise ShapeError(f"needs a concrete model width, but {t.shape} ends in '{t.shape[2]}'.")
    width, heads = t.shape[2].size, positive(cfg["num_heads"], "num_heads")
    if width % heads:
        good = [h for h in range(1, min(width, 32) + 1) if width % h == 0]
        raise ShapeError(
            f"{heads} heads do not divide a model width of {width}.",
            hint="Try " + ", ".join(str(h) for h in good[:8]) + ".",
        )
    positive(cfg["repeat"], "repeat")
    return same(t)


layer(
    "transformer_block", "Transformer Block", CAT,
    doc="Self-attention and a feed-forward, each wrapped in a residual with a layer norm. "
        "The shape never changes, so you can stack as many as you like — that is what "
        "'repeat' does, and each copy gets its own weights. Turn on 'causal' for a model "
        "that generates text.",
    aliases=("transformer", "gpt block", "bert block", "encoder layer", "attention block"),
    params=(
        P("num_heads", "int", 4, "Attention heads. Must divide the model width.", min=1, max=64),
        P("repeat", "int", 1, "How many blocks in a row. Each gets its own weights.", min=1, max=64),
        P("ff_mult", "float", 4.0, "The feed-forward is this many times wider than the model.",
          min=0.25, max=16.0, step=0.25),
        P("causal", "bool", False, "Stop each position from reading later ones."),
        P("dropout", "float", 0.0, "Dropout inside the block.", min=0.0, max=0.9, step=0.05),
        P("activation", "combo", "gelu", "Activation inside the feed-forward.", choices=_ACTS),
    ),
    build=lambda s, c: modules.TransformerStack(
        s[0][2].size, num_heads=positive(c["num_heads"], "num_heads"), ff_mult=float(c["ff_mult"]),
        dropout=float(c["dropout"]), causal=bool(c["causal"]), activation=str(c["activation"]),
        repeat=positive(c["repeat"], "repeat")),
    emit_init=lambda s, c: "TransformerStack({}, {})".format(
        s[0][2].size,
        kwargs_src(num_heads=int(c["num_heads"]), ff_mult=float(c["ff_mult"]),
                   dropout=float(c["dropout"]), causal=bool(c["causal"]),
                   activation=str(c["activation"]), repeat=int(c["repeat"]))),
    helpers=(_ACTS_SRC, inspect.getsource(modules.causal_mask),
             inspect.getsource(modules.TransformerBlock),
             inspect.getsource(modules.TransformerStack)),
)(_transformer_infer)
