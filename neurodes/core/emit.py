"""Printing the trace as PyTorch source.

This is the point of the whole exercise. Dragging boxes is a fine way to think, but it is
a dead end if the result only runs inside the tool that drew it. What comes out of here is
an ordinary file: readable, editable, dependency-free apart from torch, and identical in
behaviour to the model the workflow just trained.

Every forward line is annotated with the shape it produces, which makes the exported file
a better tutorial than most.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

from .registry import get
from .shape import Shape
from .trace import Op, SymTensor, assign_names, graph_inputs, topo_ops

_HEADER = '''"""{name} — exported from a neurodes workflow.

{summary}

Every line in forward() is annotated with the shape it produces. Symbolic names such as B
stand for sizes that are decided at run time.
"""

import torch
import torch.nn.functional as F
from torch import nn
'''


def _identifier(text: str, fallback: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in str(text)).strip("_")
    if not out or out[0].isdigit():
        out = fallback
    return out


def _class_name(name: str) -> str:
    parts = [p for p in _identifier(name, "Model").split("_") if p]
    joined = "".join(p[:1].upper() + p[1:] for p in parts)
    return joined or "Model"


def emit_source(outputs: Sequence[SymTensor], name: str = "Model",
                include_demo: bool = True) -> str:
    """Render the graph reachable from ``outputs`` as a complete Python module."""
    outs = list(outputs)
    ops = topo_ops(outs)
    inputs = graph_inputs(outs)
    names = assign_names(ops)
    out_shapes = shapes_by_op(outs)
    cls = _class_name(name)

    helpers: list[str] = []
    init_lines: list[str] = []
    forward_lines: list[str] = []
    var: dict[int, str] = {}
    used_vars: set[str] = set()
    emitted_init: set[str] = set()

    for op in inputs:
        v = _unique(_identifier(op.params.get("name", "x"), "x"), used_vars)
        var[op.uid] = v

    for op in ops:
        if op.kind == "input":
            continue
        spec = get(op.kind)
        shapes = [t.shape for t in op.inputs]
        cfg = spec.runtime_cfg(shapes, op.params)
        attr = names[op.uid]

        for helper in spec.helpers:
            block = textwrap.dedent(helper).strip("\n")
            if block not in helpers:
                helpers.append(block)

        if attr not in emitted_init:
            rhs = spec.source_init(shapes, op.params)
            if rhs is not None:
                shared = f"  # shared: {op.share}" if op.share else ""
                init_lines.append(f"self.{attr} = {rhs}{shared}")
            emitted_init.add(attr)

        args = [var[t.producer.uid] for t in op.inputs]
        expr = spec.source_call(attr, args, cfg)
        v = _unique(attr, used_vars)
        var[op.uid] = v
        forward_lines.append((f"{v} = {expr}", str(out_shapes[op.uid])))

    ret = ", ".join(var[t.producer.uid] for t in outs)
    arg_list = ", ".join(var[op.uid] for op in inputs)

    body_init = "\n".join(f"        {line}" for line in init_lines) or "        pass"
    width = max((len(line) for line, _ in forward_lines), default=0)
    body_forward = "\n".join(
        f"        {line}{' ' * (width - len(line))}  # {shape}" for line, shape in forward_lines
    ) or "        pass"

    summary = _summary_line(inputs, outs, var)
    parts = [_HEADER.format(name=cls, summary=summary)]
    if helpers:
        parts.append("\n\n" + "\n\n\n".join(helpers) + "\n")
    parts.append(f"""

class {cls}(nn.Module):
    def __init__(self):
        super().__init__()
{body_init}

    def forward(self, {arg_list}):
{body_forward}
        return {ret}
""")

    if include_demo:
        parts.append(_demo(cls, inputs, var))
    return "".join(parts)


def _unique(stem: str, used: set[str]) -> str:
    candidate = stem
    n = 2
    while candidate in used:
        candidate = f"{stem}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def shapes_by_op(outs: Sequence[SymTensor]) -> dict[int, Shape]:
    """Map every op to the shape it produced, by collecting the tensors already in the graph.

    Every op's output is either a model output or an input to something downstream, so one
    walk finds all of them — no need to re-run inference.
    """
    found: dict[int, Shape] = {}
    stack = list(outs)
    seen: set[int] = set()
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if t.producer is not None:
            found[t.producer.uid] = t.shape
            stack.extend(t.producer.inputs)
    return found


def _summary_line(inputs: Sequence[Op], outs: Sequence[SymTensor], var: dict[int, str]) -> str:
    ins = ", ".join(f"{var[op.uid]}: {op.params.get('shape', '?')}" for op in inputs)
    outs_txt = ", ".join(str(t.shape) for t in outs)
    return f"Takes {ins}\nReturns {outs_txt}"


def _demo(cls: str, inputs: Sequence[Op], var: dict[int, str]) -> str:
    made = []
    for op in inputs:
        shape = Shape.parse(op.params.get("shape", "1"))
        sizes = ", ".join(str(d.size) if d.is_concrete else "8" for d in shape.dims)
        dtype = op.params.get("dtype", "float32")
        if dtype == "int64":
            made.append(f"    {var[op.uid]} = torch.randint(0, 10, ({sizes},))")
        elif dtype == "bool":
            made.append(f"    {var[op.uid]} = torch.zeros(({sizes},), dtype=torch.bool)")
        else:
            made.append(f"    {var[op.uid]} = torch.randn({sizes})")
    call = ", ".join(var[op.uid] for op in inputs)
    body = "\n".join(made)
    return f'''

if __name__ == "__main__":
    model = {cls}()
    total = sum(p.numel() for p in model.parameters())
    print(f"{{total:,}} parameters")

{body}
    out = model({call})
    print("output:", out.shape if torch.is_tensor(out) else [o.shape for o in out])
'''
