"""The symbolic graph that travels along the wires.

Nothing here computes anything. A ``SymTensor`` is "the data at this point in the network"
reduced to what you can know before training: its shape and dtype, plus a back-pointer to
the operation that produced it. Because each tensor points back at its producer, and each
producer points back at *its* inputs, the whole architecture is reachable from the output
tensor alone. That is exactly the structure ComfyUI's wires already have, so no shared
mutable graph object is needed and node execution stays a pure function.

The trade is that the ComfyUI graph is *traced*, not *run*: dropping a Linear node on a
wire records "a Linear happens here" and computes the resulting shape. Real ``nn.Module``
objects appear once, later, in :mod:`neurodes.core.compile`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .errors import GraphError
from .shape import Shape

_op_ids = itertools.count(1)

#: dtypes a neurodes tensor may carry. Kept small on purpose.
DTYPES = ("float32", "float16", "bfloat16", "int64", "bool")

FLOAT_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(eq=False)
class Op:
    """One operation in the traced network."""

    kind: str
    """Registry key, e.g. ``"linear"``, ``"conv2d"``, ``"add"``."""

    params: dict[str, Any] = field(default_factory=dict)
    inputs: tuple["SymTensor", ...] = ()
    share: str = ""
    """Non-empty means: every op with this tag uses one shared set of weights."""

    label: str = ""
    """Optional user label, used for the variable name in exported code."""

    uid: int = field(default_factory=lambda: next(_op_ids))

    def __hash__(self) -> int:
        return self.uid

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<Op {self.kind}#{self.uid}>"

    @property
    def parents(self) -> tuple["Op", ...]:
        return tuple(t.producer for t in self.inputs if t.producer is not None)

    def config_key(self) -> tuple:
        """Everything that determines the weights of this op, for share-tag checking."""
        return (self.kind, tuple(sorted((k, _freeze(v)) for k, v in self.params.items())),
                tuple(str(t.shape) for t in self.inputs))


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


@dataclass(eq=False, frozen=True)
class SymTensor:
    """A tensor as it is known at design time: a shape, a dtype, and where it came from."""

    shape: Shape
    dtype: str = "float32"
    producer: Op | None = None
    index: int = 0

    def __hash__(self) -> int:
        return id(self)

    @property
    def rank(self) -> int:
        return self.shape.rank

    @property
    def is_float(self) -> bool:
        return self.dtype in FLOAT_DTYPES

    def describe(self) -> str:
        base = str(self.shape)
        return base if self.dtype == "float32" else f"{base} {self.dtype}"

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        src = self.producer.kind if self.producer else "?"
        return f"<SymTensor {self.shape} {self.dtype} from {src}>"


def make_input(shape: Shape, dtype: str = "float32", name: str = "x") -> SymTensor:
    """Create a network entry point."""
    if dtype not in DTYPES:
        raise GraphError(
            f"Unknown dtype {dtype!r}.",
            hint="Pick one of: " + ", ".join(DTYPES),
        )
    op = Op(kind="input", params={"name": name, "shape": str(shape), "dtype": dtype}, label=name)
    return SymTensor(shape=shape, dtype=dtype, producer=op)


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def topo_ops(outputs: Sequence[SymTensor]) -> list[Op]:
    """Every op reachable from ``outputs``, parents before children.

    Deterministic: the order depends only on the graph structure and the order of
    ``outputs``, never on the order the nodes happened to execute in.
    """
    order: list[Op] = []
    seen: set[int] = set()

    def visit(op: Op) -> None:
        if op.uid in seen:
            return
        seen.add(op.uid)
        for parent in op.parents:
            visit(parent)
        order.append(op)

    for t in outputs:
        if t.producer is None:
            raise GraphError(
                "An output tensor has no producer.",
                hint="Every output must come from a node. Connect the output to a layer.",
            )
        visit(t.producer)
    return order


def graph_inputs(outputs: Sequence[SymTensor]) -> list[Op]:
    """The input ops the outputs actually depend on, in traversal order."""
    return [op for op in topo_ops(outputs) if op.kind == "input"]


def assign_names(ops: Iterable[Op]) -> dict[int, str]:
    """A readable, unique, deterministic Python identifier per op.

    Shared ops (same ``share`` tag) get one name, because they are one module.
    """
    names: dict[int, str] = {}
    by_share: dict[str, str] = {}
    counters: dict[str, int] = {}

    for op in ops:
        if op.share and op.share in by_share:
            names[op.uid] = by_share[op.share]
            continue
        stem = _identifier(op.label) if op.label else op.kind
        counters[stem] = counters.get(stem, 0) + 1
        name = f"{stem}_{counters[stem]}"
        names[op.uid] = name
        if op.share:
            by_share[op.share] = name
    return names


def _identifier(text: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in text).strip("_").lower()
    if not out or out[0].isdigit():
        out = "n_" + out
    return out


def share_groups(ops: Sequence[Op]) -> dict[str, list[Op]]:
    """Group ops by share tag and verify each group is genuinely shareable."""
    groups: dict[str, list[Op]] = {}
    for op in ops:
        if op.share:
            groups.setdefault(op.share, []).append(op)
    for tag, members in groups.items():
        first = members[0].config_key()
        for other in members[1:]:
            if other.config_key() != first:
                raise GraphError(
                    f"Layers sharing weights under the tag '{tag}' are not identical: "
                    f"{members[0].kind} and {other.kind} differ in configuration or input shape.",
                    hint="Weight sharing needs the same layer type, the same settings and the "
                         "same input shape. Change the tag on one of them if they are meant "
                         "to be separate layers.",
                )
    return groups


def count_ops(outputs: Sequence[SymTensor]) -> int:
    return len(topo_ops(outputs))
