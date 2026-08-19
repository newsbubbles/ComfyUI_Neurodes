"""Attention, and the position information it needs to work at all."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from ..errors import ShapeError
from ..registry import P, layer, positive, require_float
from ..shape import Dim, Shape, unify_dim
from . import modules
from ._common import kwargs_src, out, same

CAT = "layers/attention"

_MASK_SRC = inspect.getsource(modules.causal_mask)


def _seq_check(t, display: str, what: str = "[batch, time, features]"):
    require_float(t, display)
    if t.rank != 3:
        raise ShapeError(
            f"expects {what}, but got {t.shape} (rank {t.rank}).",
            hint="Attention works over a sequence. If you have images, Flatten the spatial "
                 "dimensions into a time axis first, or use a convolution instead.",
        )
    if not t.shape[2].is_concrete:
        raise ShapeError(
            f"needs a concrete feature width, but {t.shape} ends in '{t.shape[2]}'.",
            hint="Give the model dimension a number.",
        )


def _heads_check(width: int, heads: int, display: str):
    if width % heads:
        good = [h for h in range(1, min(width, 32) + 1) if width % h == 0]
        raise ShapeError(
            f"{display}: {heads} heads do not divide a width of {width}.",
            hint="Each head gets an equal slice of the features. Try "
                 + ", ".join(str(h) for h in good[:8]) + ".",
        )


# ---------------------------------------------------------------------------
# Self attention
# ---------------------------------------------------------------------------

def _self_attn_infer(ins, cfg):
    t = ins[0]
    _seq_check(t, "Self Attention")
    _heads_check(t.shape[2].size, positive(cfg["num_heads"], "num_heads"), "Self Attention")
    return same(t)


def _self_attn_apply(m, ts, c):
    x = ts[0]
    mask = modules.causal_mask(x.shape[1], x.device) if bool(c["causal"]) else None
    return m(x, x, x, attn_mask=mask, need_weights=False)[0]


def _self_attn_emit_call(attr, args, c):
    x = args[0]
    mask = f"causal_mask({x}.shape[1], {x}.device)" if bool(c["causal"]) else "None"
    return f"self.{attr}({x}, {x}, {x}, attn_mask={mask}, need_weights=False)[0]"


layer(
    "self_attention", "Self Attention", CAT,
    doc="Every position looks at every other position and pulls in what it finds useful. "
        "The shape does not change; what changes is that each position now knows about its "
        "context. Turn on 'causal' when the model must not read ahead, which is what makes "
        "a language model able to generate.",
    aliases=("multihead attention", "mha", "transformer", "attend", "qkv"),
    params=(
        P("num_heads", "int", 4, "How many independent attention patterns run in parallel. "
                                 "Must divide the feature width.", min=1, max=64),
        P("causal", "bool", False, "Stop each position from seeing later ones."),
        P("dropout", "float", 0.0, "Dropout on the attention weights.", min=0.0, max=0.9, step=0.05),
    ),
    build=lambda s, c: nn.MultiheadAttention(
        s[0][2].size, positive(c["num_heads"], "num_heads"),
        dropout=float(c["dropout"]), batch_first=True),
    apply=_self_attn_apply,
    emit_init=lambda s, c: "nn.MultiheadAttention({}, {}, {})".format(
        s[0][2].size, int(c["num_heads"]),
        kwargs_src(dropout=float(c["dropout"]), batch_first=True)),
    emit_call=_self_attn_emit_call,
    helpers=(_MASK_SRC,),
)(_self_attn_infer)


# ---------------------------------------------------------------------------
# Cross attention
# ---------------------------------------------------------------------------

def _cross_attn_infer(ins, cfg):
    q, ctx = ins[0], ins[1]
    _seq_check(q, "Cross Attention")
    _seq_check(ctx, "Cross Attention")
    try:
        unify_dim(q.shape[2], ctx.shape[2])
    except ShapeError:
        raise ShapeError(
            f"the query width {q.shape[2]} and the context width {ctx.shape[2]} differ.",
            hint="Put a Linear on one of them so both end in the same number of features.",
        ) from None
    _heads_check(q.shape[2].size, positive(cfg["num_heads"], "num_heads"), "Cross Attention")
    return same(q)


layer(
    "cross_attention", "Cross Attention", CAT,
    doc="One sequence looks at a different one. The query decides what it wants, the context "
        "supplies it. This is how a caption reads an image, or how a decoder consults an "
        "encoder. The output has the query's length and the query's width.",
    aliases=("encoder decoder attention", "conditioning", "context"),
    arity=2,
    input_names=("query", "context"),
    params=(
        P("num_heads", "int", 4, "How many attention patterns run in parallel.", min=1, max=64),
        P("dropout", "float", 0.0, "Dropout on the attention weights.", min=0.0, max=0.9, step=0.05),
    ),
    build=lambda s, c: nn.MultiheadAttention(
        s[0][2].size, positive(c["num_heads"], "num_heads"),
        dropout=float(c["dropout"]), batch_first=True),
    apply=lambda m, ts, c: m(ts[0], ts[1], ts[1], need_weights=False)[0],
    emit_init=lambda s, c: "nn.MultiheadAttention({}, {}, {})".format(
        s[0][2].size, int(c["num_heads"]),
        kwargs_src(dropout=float(c["dropout"]), batch_first=True)),
    emit_call=lambda attr, args, c: f"self.{attr}({args[0]}, {args[1]}, {args[1]}, need_weights=False)[0]",
)(_cross_attn_infer)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def _positions_infer(display: str):
    def infer(ins, cfg):
        t = ins[0]
        _seq_check(t, display)
        length = t.shape[1]
        limit = positive(cfg["max_len"], "max_len")
        if length.is_concrete and length.size > limit:
            raise ShapeError(
                f"{display}: the sequence is {length.size} long but max_len is {limit}.",
                hint=f"Raise max_len to at least {length.size}.",
            )
        return same(t)
    return infer


layer(
    "sinusoidal_positions", "Sinusoidal Positions", CAT,
    doc="Adds a fixed pattern of sine waves so the model can tell positions apart. Attention "
        "on its own is order-blind — shuffle the input and you get the shuffled output — so "
        "without something like this a transformer cannot tell a sentence from its anagram. "
        "Learns nothing, and works on sequences longer than it saw in training.",
    aliases=("positional encoding", "pe", "position"),
    params=(P("max_len", "int", 4096, "The longest sequence this will ever see.", min=1, max=1 << 20),),
    build=lambda s, c: modules.SinusoidalPositions(s[0][2].size, positive(c["max_len"], "max_len")),
    emit_init=lambda s, c: f"SinusoidalPositions({s[0][2].size}, max_len={int(c['max_len'])})",
    helpers=(inspect.getsource(modules.SinusoidalPositions),),
    trains=False,
)(_positions_infer("Sinusoidal Positions"))


layer(
    "learned_positions", "Learned Positions", CAT,
    doc="A position embedding the network trains for itself. Usually a little better than "
        "the sinusoidal one inside its trained length, and useless past it.",
    aliases=("positional embedding", "learned pe"),
    params=(P("max_len", "int", 1024, "The longest sequence this will ever see.", min=1, max=1 << 20),),
    build=lambda s, c: modules.LearnedPositions(s[0][2].size, positive(c["max_len"], "max_len")),
    emit_init=lambda s, c: f"LearnedPositions({s[0][2].size}, max_len={int(c['max_len'])})",
    helpers=(inspect.getsource(modules.LearnedPositions),),
)(_positions_infer("Learned Positions"))
