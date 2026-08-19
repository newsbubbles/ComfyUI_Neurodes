"""Recurrent layers — networks that read a sequence one step at a time and carry state.

torch returns ``(outputs, hidden_state)`` from these, which is the first thing that trips
people up. The node exposes the Keras choice instead: either you want the whole sequence
back, or you want the single vector the network ended up with.
"""

from __future__ import annotations

import torch
from torch import nn

from ..errors import ShapeError
from ..registry import P, layer, positive, require_float
from ..shape import Dim, Shape
from ._common import kwargs_src, out

CAT = "layers/recurrent"

_CLASSES = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
_NAMES = {"lstm": "LSTM", "gru": "GRU", "rnn": "RNN"}


def _make(kind: str, doc: str, aliases: tuple[str, ...]):
    display = _NAMES[kind]

    def infer(ins, cfg):
        t = ins[0]
        require_float(t, display)
        if t.rank != 3:
            raise ShapeError(
                f"expects [batch, time, features], but got {t.shape} (rank {t.rank}).",
                hint="A recurrent layer reads a sequence, so it needs a time axis. An "
                     "Embedding turns [batch, time] token ids into exactly this shape.",
            )
        if not t.shape[2].is_concrete:
            raise ShapeError(
                f"needs a concrete feature width, but {t.shape} ends in '{t.shape[2]}'.",
                hint="Give the last dimension a number.",
            )
        hidden = positive(cfg["hidden_size"], "hidden_size")
        width = hidden * (2 if bool(cfg["bidirectional"]) else 1)
        if bool(cfg["return_sequences"]):
            return out(Shape([t.shape[0], t.shape[1], Dim(width)]), t.dtype)
        return out(Shape([t.shape[0], Dim(width)]), t.dtype)

    def build(s, c):
        return _CLASSES[kind](
            s[0][2].size, positive(c["hidden_size"], "hidden_size"),
            num_layers=positive(c["num_layers"], "num_layers"),
            batch_first=True, bidirectional=bool(c["bidirectional"]),
            dropout=float(c["dropout"]) if int(c["num_layers"]) > 1 else 0.0,
        )

    def apply(m, ts, c):
        outputs, _ = m(ts[0])
        return outputs if bool(c["return_sequences"]) else outputs[:, -1]

    def emit_init(s, c):
        return "nn.{}({}, {}, {})".format(
            display, s[0][2].size, int(c["hidden_size"]),
            kwargs_src(num_layers=int(c["num_layers"]), batch_first=True,
                       bidirectional=bool(c["bidirectional"]),
                       dropout=float(c["dropout"]) if int(c["num_layers"]) > 1 else 0.0))

    def emit_call(attr, args, c):
        expr = f"self.{attr}({args[0]})[0]"
        return expr if bool(c["return_sequences"]) else f"{expr}[:, -1]"

    layer(
        kind, display, CAT, doc=doc, aliases=aliases,
        params=(
            P("hidden_size", "int", 128, "How much the layer remembers at each step.", min=1, max=1 << 16),
            P("num_layers", "int", 1, "How many recurrent layers to stack.", min=1, max=16),
            P("bidirectional", "bool", False, "Also read the sequence backwards, doubling the output width."),
            P("return_sequences", "bool", False,
              "On: one output per timestep, for stacking or per-token prediction. "
              "Off: only the final state, for classifying the whole sequence."),
            P("dropout", "float", 0.0, "Dropout between stacked layers. Needs num_layers above 1.",
              min=0.0, max=0.9, step=0.05, advanced=True),
        ),
        build=build, apply=apply, emit_init=emit_init, emit_call=emit_call,
    )(infer)


_make("lstm", "Reads a sequence step by step, keeping a memory it decides what to write to "
              "and what to forget. The gates are what let it hold on to something from far "
              "back in the sequence.",
      ("long short term memory", "sequence", "recurrent", "time series"))

_make("gru", "A lighter LSTM with two gates instead of three. Usually just as good and "
             "noticeably faster.",
      ("gated recurrent unit", "sequence", "recurrent"))

_make("rnn", "The plain recurrent layer, with no gates. Simple enough to reason about, and "
             "a good way to see for yourself why the gated ones exist: it forgets fast and "
             "its gradients vanish on long sequences.",
      ("simple rnn", "elman", "vanilla rnn"))
