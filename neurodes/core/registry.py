"""The single source of truth for every layer.

A layer is declared **once** here, and four consumers read that one declaration:

1. shape inference, which runs while you edit the graph;
2. the ``nn.Module`` builder, which runs when you build the model;
3. the PyTorch source emitter, which prints the file you would have written by hand;
4. the ComfyUI node generator, which turns the parameter list into widgets.

Declaring it once is the only way to guarantee that the shape the node shows you and the
module that actually gets built cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .errors import NeurodesError, ShapeError
from .shape import Shape
from .trace import FLOAT_DTYPES, Op, SymTensor

VARIADIC = -1

#: Widget kinds a parameter may use. Mapped to ComfyUI io.* inputs by the node adapter.
PARAM_KINDS = ("int", "float", "bool", "string", "combo", "ints")


@dataclass(frozen=True)
class Param:
    """One knob on a layer."""

    name: str
    kind: str = "int"
    default: Any = 0
    doc: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    advanced: bool = False
    multiline: bool = False

    def __post_init__(self):
        if self.kind not in PARAM_KINDS:
            raise NeurodesError(f"Unknown param kind {self.kind!r} on {self.name!r}")
        if self.kind == "combo" and not self.choices:
            raise NeurodesError(f"Combo param {self.name!r} needs choices")

    @property
    def display(self) -> str:
        return self.name.replace("_", " ")


def P(name: str, kind: str = "int", default: Any = 0, doc: str = "", **kw) -> Param:
    """Terse constructor, because these are written a hundred times."""
    return Param(name=name, kind=kind, default=default, doc=doc, **kw)


InferFn = Callable[[list[SymTensor], dict], SymTensor]
BuildFn = Callable[[list[Shape], dict], Any]
ApplyFn = Callable[[Any, list[Any], dict], Any]
EmitInitFn = Callable[[list[Shape], dict], str]
EmitCallFn = Callable[[str, list[str], dict], str]


@dataclass
class LayerSpec:
    """Everything neurodes knows about one kind of layer."""

    key: str
    display: str
    category: str
    infer: InferFn
    doc: str = ""
    params: tuple[Param, ...] = ()
    arity: int = 1
    """Number of tensor inputs, or :data:`VARIADIC` for "two or more"."""

    input_names: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    build: BuildFn | None = None
    """Returns an ``nn.Module``. ``None`` means the layer has no state (Add, Reshape, ...)."""

    apply: ApplyFn | None = None
    """How to run it. Default is ``module(*tensors)``; stateless layers must supply this."""

    prepare: Callable[[list[Shape], dict], dict] | None = None
    """Optional: extra config worked out once at build time, when the input shapes are known.

    Reshape and Slice need this — at run time they only see a raw tensor, and "the axis the
    user called T" is a fact about the traced graph, not about the tensor. Whatever this
    returns is merged into the config that ``apply`` and ``emit_call`` receive.
    """

    emit_init: EmitInitFn | None = None
    """Right-hand side of the ``__init__`` assignment, e.g. ``"nn.Linear(784, 128)"``."""

    emit_call: EmitCallFn | None = None
    """Forward expression. Default is ``self.<name>(<args>)``."""

    imports: tuple[str, ...] = ()
    """Extra import lines the emitted file needs for this layer."""

    helpers: tuple[str, ...] = ()
    """Module-level source (helper functions, composite ``nn.Module`` classes) the emitted
    file needs. Composite blocks fill this with ``inspect.getsource`` of the very class they
    build, so the exported file cannot drift from what actually ran."""

    trains: bool = True
    """False for layers that never hold parameters, used only for the summary table."""

    def n_inputs_label(self) -> str:
        return "any number of" if self.arity == VARIADIC else str(self.arity)

    def input_label(self, i: int) -> str:
        if i < len(self.input_names):
            return self.input_names[i]
        return "x" if self.arity == 1 else f"x{i + 1}"

    # -- the four consumers -------------------------------------------------
    def infer_output(self, inputs: list[SymTensor], cfg: dict) -> SymTensor:
        """Shape inference with the layer name attached to whatever goes wrong."""
        self.check_arity(len(inputs))
        try:
            return self.infer(inputs, cfg)
        except ShapeError as exc:
            raise ShapeError(f"{self.display}: {exc.message}", exc.hint) from None

    def check_arity(self, n: int) -> None:
        if self.arity == VARIADIC:
            if n < 2:
                raise ShapeError(
                    f"{self.display} needs at least 2 inputs, got {n}.",
                    hint="Connect another tensor to the next free input slot.",
                )
        elif n != self.arity:
            raise ShapeError(
                f"{self.display} needs exactly {self.arity} input(s), got {n}.",
                hint="Check that every input slot on this node is connected.",
            )

    def make_module(self, shapes: list[Shape], cfg: dict):
        return None if self.build is None else self.build(shapes, cfg)

    def runtime_cfg(self, shapes: list[Shape], cfg: dict) -> dict:
        """The config ``apply``/``emit_call`` should see, once shapes are known."""
        if self.prepare is None:
            return cfg
        return {**cfg, **self.prepare(shapes, cfg)}

    def run(self, module, tensors: list, cfg: dict):
        if self.apply is not None:
            return self.apply(module, tensors, cfg)
        if module is None:
            raise NeurodesError(f"{self.display} has neither a module nor an apply function.")
        return module(*tensors)

    def source_init(self, shapes: list[Shape], cfg: dict) -> str | None:
        return None if self.emit_init is None else self.emit_init(shapes, cfg)

    def source_call(self, attr: str, args: list[str], cfg: dict) -> str:
        if self.emit_call is not None:
            return self.emit_call(attr, args, cfg)
        return f"self.{attr}({', '.join(args)})"

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def normalize(self, cfg: dict) -> dict:
        """Fill in defaults and drop unknown keys, so callers can be sloppy."""
        out = self.defaults()
        for k, v in (cfg or {}).items():
            if k in out:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, LayerSpec] = {}
_ORDER: list[str] = []


def register(spec: LayerSpec) -> LayerSpec:
    if spec.key in _REGISTRY:
        raise NeurodesError(f"Layer {spec.key!r} is already registered")
    _REGISTRY[spec.key] = spec
    _ORDER.append(spec.key)
    return spec


def layer(key: str, display: str, category: str, **kw) -> Callable[[InferFn], LayerSpec]:
    """Decorator form: the decorated function is the shape-inference function."""

    def wrap(fn: InferFn) -> LayerSpec:
        spec = LayerSpec(key=key, display=display, category=category, infer=fn,
                         doc=kw.pop("doc", None) or (fn.__doc__ or "").strip(), **kw)
        return register(spec)

    return wrap


def get(key: str) -> LayerSpec:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise NeurodesError(
            f"There is no layer called {key!r}.",
            hint="Known layers: " + ", ".join(sorted(_REGISTRY)),
        ) from None


def all_specs() -> list[LayerSpec]:
    return [_REGISTRY[k] for k in _ORDER]


def categories() -> list[str]:
    seen: list[str] = []
    for spec in all_specs():
        if spec.category not in seen:
            seen.append(spec.category)
    return seen


def in_category(category: str) -> list[LayerSpec]:
    return [s for s in all_specs() if s.category == category]


# ---------------------------------------------------------------------------
# Applying a layer during tracing
# ---------------------------------------------------------------------------

def apply_layer(key: str, inputs: Sequence[SymTensor], cfg: dict | None = None,
                share: str = "", label: str = "") -> SymTensor:
    """Trace one layer: infer the output shape and record the op.

    This is what every layer node in ComfyUI ends up calling.
    """
    spec = get(key)
    tensors = list(inputs)
    for i, t in enumerate(tensors):
        if t is None:
            raise ShapeError(
                f"{spec.display}: input '{spec.input_label(i)}' is not connected.",
                hint="Drag a wire from a tensor output into that slot.",
            )
    resolved = spec.normalize(cfg or {})
    out = spec.infer_output(tensors, resolved)
    op = Op(kind=key, params=resolved, inputs=tuple(tensors), share=share.strip(), label=label.strip())
    return SymTensor(shape=out.shape, dtype=out.dtype, producer=op)


# ---------------------------------------------------------------------------
# Shared helpers used by many layer definitions
# ---------------------------------------------------------------------------

def require_float(t: SymTensor, spec_display: str) -> None:
    if t.dtype not in FLOAT_DTYPES:
        raise ShapeError(
            f"{spec_display} needs floating-point input but got {t.dtype}.",
            hint="Integer tensors are for indices. Put an Embedding or a Cast before this layer.",
        )


def require_rank(t: SymTensor, rank: int, what: str, *, at_least: bool = False) -> None:
    ok = t.rank >= rank if at_least else t.rank == rank
    if not ok:
        word = "at least " if at_least else ""
        raise ShapeError(
            f"expects {word}a rank-{rank} tensor {what}, but got {t.shape} (rank {t.rank}).",
            hint=_rank_hint(t.rank, rank),
        )


def _rank_hint(have: int, want: int) -> str:
    if have > want:
        return (f"Use Flatten to collapse the extra dimensions, or Reshape to go from "
                f"rank {have} to rank {want}.")
    return f"Use Unsqueeze or Reshape to go from rank {have} to rank {want}."


def as_ints(value: Any, n: int, name: str) -> tuple[int, ...]:
    """Read a size parameter written as ``3`` or ``3,5`` into an n-tuple."""
    if isinstance(value, int):
        return (value,) * n
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
    else:
        text = str(value).strip().strip("()[]")
        if not text:
            raise ShapeError(f"'{name}' is empty.", hint=f"Write a number, or {n} numbers separated by commas.")
        try:
            vals = [int(p) for p in text.replace("x", ",").split(",") if p.strip()]
        except ValueError:
            raise ShapeError(
                f"Could not read '{name}' = {value!r} as whole numbers.",
                hint=f"Write a single number like 3, or {n} numbers like " + ",".join(["3"] * n) + ".",
            ) from None
    if len(vals) == 1:
        return tuple(vals * n)
    if len(vals) != n:
        raise ShapeError(
            f"'{name}' has {len(vals)} values but this layer needs {n}.",
            hint=f"Write one number for all axes, or exactly {n} numbers.",
        )
    return tuple(vals)


def positive(value: int, name: str, spec_display: str = "") -> int:
    v = int(value)
    if v <= 0:
        prefix = f"{spec_display}: " if spec_display else ""
        raise ShapeError(
            f"{prefix}'{name}' must be at least 1, got {v}.",
            hint="Sizes and counts have to be positive whole numbers.",
        )
    return v
