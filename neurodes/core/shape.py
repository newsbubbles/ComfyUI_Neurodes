"""Shapes with named (symbolic) dimensions.

A dimension is either a concrete non-negative int, or a name like ``B`` or ``T`` standing
for a size that is not known while you are drawing the network. Batch size is the obvious
case; sequence length is the other one.

Shapes here **include the batch dimension**. Keras hides it in ``Input(shape=...)`` and
that omission is responsible for a large fraction of all beginner confusion about what a
shape actually is. Writing ``B, 1, 28, 28`` costs three characters and explains itself.

Text form is the primary interface::

    B, 3, 224, 224      an image batch
    B, T, 512           a sequence batch
    B, 10               a batch of 10-vectors
    ?, 128              an anonymous unknown leading dim
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Union

from .errors import ParseError, ShapeError

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")

#: How many anonymous dims we have handed out, used to make ``?`` unique per parse.
_anon_counter = 0


@dataclass(frozen=True)
class Dim:
    """One dimension: a concrete size, or a name."""

    value: Union[int, str]

    def __post_init__(self):
        if isinstance(self.value, bool):  # bool is an int subclass; reject early
            raise ParseError(f"Dimension cannot be a bool, got {self.value!r}")
        if isinstance(self.value, int):
            if self.value < 0:
                raise ParseError(
                    f"Dimension sizes cannot be negative, got {self.value}.",
                    hint="Use a name like 'B' for a size that is not known yet.",
                )
        elif isinstance(self.value, str):
            if not _NAME_RE.match(self.value):
                raise ParseError(
                    f"{self.value!r} is not a usable dimension name.",
                    hint="Names must start with a letter or underscore, e.g. B, T, seq_len.",
                )
        else:
            raise ParseError(f"A dimension must be an int or a name, got {type(self.value).__name__}")

    # -- predicates ---------------------------------------------------------
    @property
    def is_symbolic(self) -> bool:
        return isinstance(self.value, str)

    @property
    def is_concrete(self) -> bool:
        return isinstance(self.value, int)

    @property
    def size(self) -> int:
        """The concrete size. Raises if symbolic."""
        if not self.is_concrete:
            raise ShapeError(
                f"Dimension '{self.value}' has no concrete size.",
                hint="This dimension is symbolic. Give it a number, or use a layer that "
                     "does not need to know this dimension at build time.",
            )
        return int(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"Dim({self.value!r})"


def dim(value: Union[int, str, Dim]) -> Dim:
    """Coerce to a :class:`Dim`."""
    return value if isinstance(value, Dim) else Dim(value)


def _fresh_anon() -> Dim:
    global _anon_counter
    _anon_counter += 1
    return Dim(f"_anon{_anon_counter}")


@dataclass(frozen=True)
class Shape:
    """An ordered tuple of dimensions."""

    dims: tuple[Dim, ...]

    def __init__(self, dims: Iterable[Union[int, str, Dim]] = ()):
        object.__setattr__(self, "dims", tuple(dim(d) for d in dims))

    # -- construction -------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Shape":
        """Parse ``"B, 3, 224, 224"``. Also accepts ``x``/space separators and brackets."""
        if text is None:
            raise ParseError("Empty shape.", hint="Try something like 'B, 3, 224, 224'.")
        cleaned = text.strip().strip("[]()").strip()
        if not cleaned:
            raise ParseError(
                "Empty shape.",
                hint="Type the dimensions separated by commas, e.g. 'B, 3, 224, 224'. "
                     "Use a name like B for the batch size.",
            )
        parts = [p for p in re.split(r"[,\sx×]+", cleaned) if p]
        out: list[Dim] = []
        for p in parts:
            if p == "?":
                out.append(_fresh_anon())
            elif re.fullmatch(r"-?\d+", p):
                n = int(p)
                if n < 0:
                    raise ParseError(
                        f"Dimension {n} is negative in shape '{text}'.",
                        hint="Use a name like B for an unknown size instead of -1.",
                    )
                out.append(Dim(n))
            else:
                try:
                    out.append(Dim(p))
                except ParseError as exc:
                    raise ParseError(
                        f"Could not read '{p}' in shape '{text}'. {exc.message}", exc.hint
                    ) from None
        return cls(out)

    # -- sequence protocol --------------------------------------------------
    @property
    def rank(self) -> int:
        return len(self.dims)

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self) -> Iterator[Dim]:
        return iter(self.dims)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Shape(self.dims[idx])
        return self.dims[idx]

    # -- derived ------------------------------------------------------------
    @property
    def is_concrete(self) -> bool:
        return all(d.is_concrete for d in self.dims)

    @property
    def symbols(self) -> tuple[str, ...]:
        seen: list[str] = []
        for d in self.dims:
            if d.is_symbolic and d.value not in seen:
                seen.append(str(d.value))
        return tuple(seen)

    def numel(self, skip_leading: int = 0) -> int:
        """Product of dims from ``skip_leading`` onward. Requires those dims concrete."""
        total = 1
        for d in self.dims[skip_leading:]:
            total *= d.size
        return total

    def replace(self, idx: int, new: Union[int, str, Dim]) -> "Shape":
        dims = list(self.dims)
        dims[normalize_axis(idx, self.rank)] = dim(new)
        return Shape(dims)

    def insert(self, idx: int, new: Union[int, str, Dim]) -> "Shape":
        dims = list(self.dims)
        dims.insert(idx if idx >= 0 else idx + self.rank + 1, dim(new))
        return Shape(dims)

    def drop(self, idx: int) -> "Shape":
        dims = list(self.dims)
        del dims[normalize_axis(idx, self.rank)]
        return Shape(dims)

    def concrete_or_placeholder(self, placeholder: int = 2) -> tuple[int, ...]:
        """Concrete sizes, substituting ``placeholder`` for every symbolic dim.

        Used to push a dummy batch through a built module to double-check inference.
        """
        return tuple(d.size if d.is_concrete else placeholder for d in self.dims)

    # -- display ------------------------------------------------------------
    def __str__(self) -> str:
        return "[" + ", ".join(str(d) for d in self.dims) + "]"

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"Shape({str(self)})"


def normalize_axis(axis: int, rank: int, *, allow_end: bool = False) -> int:
    """Turn a possibly-negative axis into a real index, with a readable failure.

    ``allow_end`` widens the range by one, for insertion positions (Unsqueeze), where
    ``axis == rank`` means "after the last dimension".
    """
    limit = rank + 1 if allow_end else rank
    a = axis + limit if axis < 0 else axis
    if not 0 <= a < limit:
        raise ShapeError(
            f"Axis {axis} is out of range for a rank-{rank} tensor.",
            hint=f"Valid axes are {-limit} to {limit - 1}.",
        )
    return a


def unify_dim(a: Dim, b: Dim, *, where: str = "", broadcast: bool = False) -> Dim:
    """Merge two dimensions that must describe the same axis.

    Concrete beats symbolic, because it carries more information. Two different concrete
    sizes, or two differently-named symbols, is an error the user has to resolve.
    """
    if a == b:
        return a
    if broadcast:
        if a.is_concrete and a.value == 1:
            return b
        if b.is_concrete and b.value == 1:
            return a
    if a.is_concrete and b.is_concrete:
        raise ShapeError(
            f"Size {a} does not match size {b}{where}.",
            hint="Both tensors must agree on this dimension"
                 + (", or one of them must be 1 so it can broadcast." if broadcast else "."),
        )
    if a.is_concrete:
        return a
    if b.is_concrete:
        return b
    raise ShapeError(
        f"Dimension names '{a}' and '{b}' do not match{where}.",
        hint=f"Rename one of them so both read '{a}', or give one a concrete size.",
    )


def unify(a: Shape, b: Shape, *, what: str = "these tensors") -> Shape:
    """Require two shapes to be the same, dimension by dimension."""
    if a.rank != b.rank:
        raise ShapeError(
            f"{what} have different ranks: {a} is rank {a.rank}, {b} is rank {b.rank}.",
            hint="Use Reshape, Flatten or Unsqueeze so both have the same number of dimensions.",
        )
    return Shape(
        [unify_dim(x, y, where=f" (axis {i})") for i, (x, y) in enumerate(zip(a.dims, b.dims))]
    )


def broadcast(shapes: Sequence[Shape], *, what: str = "these tensors") -> Shape:
    """NumPy/torch broadcasting over any number of shapes."""
    if not shapes:
        raise ShapeError("Nothing to broadcast.")
    rank = max(s.rank for s in shapes)
    padded = [Shape([Dim(1)] * (rank - s.rank) + list(s.dims)) for s in shapes]
    out: list[Dim] = []
    for axis in range(rank):
        acc = padded[0][axis]
        for other in padded[1:]:
            try:
                acc = unify_dim(acc, other[axis], where=f" (axis {axis})", broadcast=True)
            except ShapeError as exc:
                raise ShapeError(
                    f"Cannot broadcast {what}: " + exc.message.rstrip("."),
                    hint="Shapes are "
                         + " and ".join(str(s) for s in shapes)
                         + ". Broadcasting needs each axis to match, or to be 1 in one of them.",
                ) from None
        out.append(acc)
    return Shape(out)


def conv_out(size: Dim, kernel: int, stride: int, padding: int, dilation: int, *, axis_name: str) -> Dim:
    """Standard convolution output-size formula, kept symbolic when it has to be."""
    if not size.is_concrete:
        return Dim(f"{size}_out") if stride != 1 or kernel != 1 else size
    eff = dilation * (kernel - 1) + 1
    out = (size.size + 2 * padding - eff) // stride + 1
    if out <= 0:
        raise ShapeError(
            f"The {axis_name} dimension collapses to {out}: input {size}, kernel {kernel}, "
            f"stride {stride}, padding {padding}, dilation {dilation}.",
            hint="The window is larger than the input. Use a smaller kernel, more padding, "
                 "or fewer downsampling layers before this one.",
        )
    return Dim(out)


def convt_out(size: Dim, kernel: int, stride: int, padding: int, output_padding: int, dilation: int) -> Dim:
    """Transposed-convolution output size."""
    if not size.is_concrete:
        return Dim(f"{size}_up")
    out = (size.size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    if out <= 0:
        raise ShapeError(
            f"Transposed convolution produces a non-positive size ({out}) from input {size}.",
            hint="Reduce padding, or increase kernel size / stride.",
        )
    return Dim(out)
