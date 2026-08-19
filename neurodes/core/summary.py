"""The architecture table — what the network is, in one screen of text."""

from __future__ import annotations

from typing import Sequence

from .compile import parameter_count
from .emit import shapes_by_op
from .registry import get
from .trace import Op, SymTensor, assign_names, graph_inputs, topo_ops


def _human(n: int) -> str:
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= limit:
            return f"{n / limit:.2f}{suffix}".replace(".00", "")
    return str(n)


def rows(outputs: Sequence[SymTensor]) -> list[dict]:
    """One row per op: name, type, output shape, parameter count, inputs."""
    ops = topo_ops(list(outputs))
    names = assign_names(ops)
    shapes = shapes_by_op(list(outputs))
    counted: set[str] = set()
    result = []
    for op in ops:
        name = names[op.uid]
        if op.kind == "input":
            result.append({
                "name": name, "type": "Input", "shape": str(shapes[op.uid]),
                "params": 0, "inputs": [], "shared": False,
            })
            continue
        spec = get(op.kind)
        shared = bool(op.share) and name in counted
        params = 0 if shared else parameter_count(op)
        counted.add(name)
        result.append({
            "name": name, "type": spec.display, "shape": str(shapes[op.uid]),
            "params": params, "inputs": [names[t.producer.uid] for t in op.inputs],
            "shared": shared, "share_tag": op.share,
        })
    return result


def summarize(outputs: Sequence[SymTensor], name: str = "Model") -> str:
    """A fixed-width table, the way ``model.summary()`` should have looked."""
    data = rows(outputs)
    total = sum(r["params"] for r in data)
    trainable_layers = sum(1 for r in data if r["params"] > 0)

    headers = ("layer", "type", "output shape", "params", "from")
    table = [(r["name"], r["type"] + (" (shared)" if r["shared"] else ""),
              r["shape"], f"{r['params']:,}" if r["params"] else "-",
              ", ".join(r["inputs"]) or "-") for r in data]
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]

    def line(cells, fill=" "):
        return "  ".join(str(c).ljust(w, fill) for c, w in zip(cells, widths)).rstrip()

    outs = ", ".join(str(t.shape) for t in outputs)
    ins = ", ".join(str(shapes_by_op(list(outputs))[op.uid]) for op in graph_inputs(list(outputs)))

    body = [
        name,
        "=" * len(line(headers)),
        line(headers),
        line(["-" * w for w in widths], "-"),
    ]
    body += [line(row) for row in table]
    body += [
        "=" * len(line(headers)),
        f"input      {ins}",
        f"output     {outs}",
        f"layers     {len(data)} nodes, {trainable_layers} with weights",
        f"parameters {total:,}  ({_human(total)})",
        f"size       {_human_bytes(total * 4)} at float32",
    ]
    return "\n".join(body)


def _human_bytes(n: int) -> str:
    for limit, suffix in ((1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")):
        if n >= limit:
            return f"{n / limit:.2f} {suffix}"
    return f"{n} B"


def badge(tensor: SymTensor, params: int = 0) -> str:
    """The short line the ComfyUI node shows under itself after a run."""
    text = tensor.describe()
    if params:
        text += f"  ·  {_human(params)} params"
    return text
