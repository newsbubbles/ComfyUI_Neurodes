"""Datasets.

The toy 2d sets are here for a specific reason: a network trained on two-dimensional
points can have its entire learned function drawn as a picture. Being able to see the
decision boundary bend as you add a layer is worth more than any number printed to four
decimal places, and none of these need a download.

Everything is generated with torch alone. torchvision is imported lazily, only if someone
asks for MNIST or CIFAR, and its absence is reported as a plain sentence rather than an
ImportError traceback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from .errors import NeurodesError
from .runtime import adopt, allocating
from .shape import Dim, Shape


@dataclass
class DataBundle:
    """A dataset, already split, already tensors."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    task: str = "classification"
    """``classification``, ``regression``, ``reconstruction`` or ``discovery``.

    Decides the default loss, which metrics mean anything, and what the training node
    checks before it starts. ``discovery`` is the odd one out: it has no target at all, and
    what replaces the target is a statement about what a good *answer* looks like rather
    than about what the answer is.
    """

    classes: tuple[str, ...] = ()
    name: str = "dataset"
    notes: str = ""
    side_train: tuple[torch.Tensor, ...] = ()
    """Further inputs, after the first, in the order the model's Input nodes were traced.

    A model can take more than one thing: two towers comparing a pair, a picture plus the
    timestep that says how noisy it is, an image plus a mask. ``x_train`` stays the first
    input so that everything written for the single-input case keeps working, and these ride
    alongside it. Empty for almost every dataset.
    """

    side_val: tuple[torch.Tensor, ...] = ()
    extra: dict = field(default_factory=dict)
    """Anything a later node needs that is not the data itself.

    Diffusion puts its noise schedule here, so the sampler reads the settings the dataset
    was actually built with instead of asking the user to type them twice and get one of
    them wrong.
    """

    TASKS = ("classification", "regression", "reconstruction", "discovery")

    def __post_init__(self):
        if self.task not in self.TASKS:
            raise NeurodesError(f"Unknown task {self.task!r}",
                                hint="Use one of: " + ", ".join(self.TASKS))

    # -- shapes -------------------------------------------------------------
    @property
    def input_shape(self) -> Shape:
        """The shape an Input node should declare, with a symbolic batch dimension."""
        return Shape(["B"] + [int(s) for s in self.x_train.shape[1:]])

    @property
    def input_shapes(self) -> list[Shape]:
        """One shape per input, in the order a model has to accept them."""
        return [Shape(["B"] + [int(s) for s in t.shape[1:]])
                for t in (self.x_train, *self.side_train)]

    @property
    def n_inputs(self) -> int:
        return 1 + len(self.side_train)

    @property
    def train_inputs(self) -> tuple[torch.Tensor, ...]:
        return (self.x_train, *self.side_train)

    @property
    def val_inputs(self) -> tuple[torch.Tensor, ...]:
        return (self.x_val, *self.side_val)

    @property
    def target_shape(self) -> Shape:
        return Shape(["B"] + [int(s) for s in self.y_train.shape[1:]])

    @property
    def n_classes(self) -> int:
        if self.task != "classification":
            return 0
        return len(self.classes) if self.classes else int(self.y_train.max().item()) + 1

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.x_val.shape[0])

    @property
    def is_2d_points(self) -> bool:
        """Can this be drawn as a decision boundary?"""
        return (self.task == "classification" and self.x_train.dim() == 2
                and self.x_train.shape[1] == 2)

    def describe(self) -> str:
        lines = [
            f"{self.name}",
            f"  task        {self.task}",
            f"  train       {self.n_train} examples",
            f"  validation  {self.n_val} examples",
            f"  input       {self.input_shape}",
        ]
        for i, shape in enumerate(self.input_shapes[1:], start=2):
            lines.append(f"  input {i}     {shape}")
        lines.append("  target      none — nothing here says what the answer should be"
                     if self.task == "discovery" else f"  target      {self.target_shape}")
        if self.task == "classification":
            lines.append(f"  classes     {self.n_classes}" +
                         (f"  ({', '.join(self.classes)})" if self.classes else ""))
        elif self.task == "reconstruction":
            same = (self.y_train.shape == self.x_train.shape
                    and torch.equal(self.x_train, self.y_train))
            lines.append("  target      " + ("the input itself" if same else "a paired image"))
        if self.notes:
            lines.append(f"  note        {self.notes}")
        return "\n".join(lines)

    def to(self, device) -> "DataBundle":
        return DataBundle(
            self.x_train.to(device), self.y_train.to(device),
            self.x_val.to(device), self.y_val.to(device),
            self.task, self.classes, self.name, self.notes,
            tuple(t.to(device) for t in self.side_train),
            tuple(t.to(device) for t in self.side_val),
            dict(self.extra),
        )


# ---------------------------------------------------------------------------
# Synthetic 2d shapes
# ---------------------------------------------------------------------------

def _split(x: torch.Tensor, y: torch.Tensor, val_fraction: float, generator):
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator)
    cut = max(1, int(n * (1.0 - val_fraction)))
    return x[perm[:cut]], y[perm[:cut]], x[perm[cut:]], y[perm[cut:]]


def _moons(n: int, noise: float, g):
    half = n // 2
    t = torch.rand(half, generator=g) * math.pi
    outer = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    t2 = torch.rand(n - half, generator=g) * math.pi
    inner = torch.stack([1 - torch.cos(t2), 0.5 - torch.sin(t2)], dim=1)
    x = torch.cat([outer, inner]) + torch.randn(n, 2, generator=g) * noise
    y = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(n - half, dtype=torch.long)])
    return x, y, ("outer", "inner")


def _circles(n: int, noise: float, g):
    half = n // 2
    a1 = torch.rand(half, generator=g) * 2 * math.pi
    a2 = torch.rand(n - half, generator=g) * 2 * math.pi
    inner = torch.stack([torch.cos(a1), torch.sin(a1)], dim=1) * 0.4
    outer = torch.stack([torch.cos(a2), torch.sin(a2)], dim=1) * 1.0
    x = torch.cat([inner, outer]) + torch.randn(n, 2, generator=g) * noise
    y = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(n - half, dtype=torch.long)])
    return x, y, ("inside", "outside")


def _spirals(n: int, noise: float, g, arms: int = 3):
    per = n // arms
    xs, ys = [], []
    for k in range(arms):
        t = torch.linspace(0.2, 1.0, per) * 3.5
        angle = t * 2.2 + k * (2 * math.pi / arms)
        pts = torch.stack([t * torch.cos(angle), t * torch.sin(angle)], dim=1) / 3.5
        xs.append(pts + torch.randn(per, 2, generator=g) * noise)
        ys.append(torch.full((per,), k, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys), tuple(f"arm {k}" for k in range(arms))


def _blobs(n: int, noise: float, g, k: int = 3):
    per = n // k
    centres = torch.stack([
        torch.tensor([math.cos(2 * math.pi * i / k), math.sin(2 * math.pi * i / k)])
        for i in range(k)
    ])
    xs = [centres[i] + torch.randn(per, 2, generator=g) * (noise + 0.15) for i in range(k)]
    ys = [torch.full((per,), i, dtype=torch.long) for i in range(k)]
    return torch.cat(xs), torch.cat(ys), tuple(f"blob {i}" for i in range(k))


def _xor(n: int, noise: float, g):
    x = (torch.rand(n, 2, generator=g) * 2 - 1)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).long()
    return x + torch.randn(n, 2, generator=g) * noise, y, ("same sign", "different sign")


def _checkerboard(n: int, noise: float, g):
    x = (torch.rand(n, 2, generator=g) * 4 - 2)
    y = ((torch.floor(x[:, 0]) + torch.floor(x[:, 1])) % 2).long()
    return x + torch.randn(n, 2, generator=g) * noise, y, ("black", "white")


TOY_SHAPES: dict[str, Callable] = {
    "two moons": _moons,
    "circles": _circles,
    "spirals": _spirals,
    "blobs": _blobs,
    "xor": _xor,
    "checkerboard": _checkerboard,
}

_TOY_NOTES = {
    "xor": "No straight line can separate these. A network with no hidden layer will sit "
           "at about 50% forever, which is the point.",
    "two moons": "Almost separable, but only by a curve.",
    "spirals": "Needs real depth. A shallow network gets the middle and loses the tails.",
    "checkerboard": "Highly non-linear. A good test of whether a network has enough capacity.",
    "circles": "One class completely surrounds the other.",
    "blobs": "Nearly linearly separable — a single Linear layer will do well.",
}


def toy_classification(shape: str = "two moons", n: int = 1000, noise: float = 0.1,
                       val_fraction: float = 0.25, seed: int = 0) -> DataBundle:
    """Two-dimensional points in a named arrangement, ready to be drawn."""
    if shape not in TOY_SHAPES:
        raise NeurodesError(f"Unknown dataset shape {shape!r}",
                            hint="Choose one of: " + ", ".join(TOY_SHAPES))
    with allocating():
        g = torch.Generator().manual_seed(int(seed))
        x, y, classes = TOY_SHAPES[shape](max(int(n), 8), float(noise), g)
        xt, yt, xv, yv = _split(x.float(), y, val_fraction, g)
    return DataBundle(xt, yt, xv, yv, "classification", classes,
                      name=f"{shape} ({len(x)} points)", notes=_TOY_NOTES.get(shape, ""))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

_CURVES = {
    "sine": lambda t: torch.sin(t * 3),
    "damped sine": lambda t: torch.sin(t * 6) * torch.exp(-t.abs()),
    "polynomial": lambda t: 0.5 * t ** 3 - t,
    "step": lambda t: torch.sign(t) * 0.7,
    "absolute": lambda t: t.abs() - 0.5,
}


def toy_regression(curve: str = "sine", n: int = 600, noise: float = 0.05,
                   val_fraction: float = 0.25, seed: int = 0) -> DataBundle:
    """One input, one output: fit a curve, then draw what the network learned."""
    if curve not in _CURVES:
        raise NeurodesError(f"Unknown curve {curve!r}", hint="Choose one of: " + ", ".join(_CURVES))
    with allocating():
        g = torch.Generator().manual_seed(int(seed))
        x = (torch.rand(max(int(n), 8), 1, generator=g) * 4 - 2)
        y = _CURVES[curve](x) + torch.randn_like(x) * float(noise)
        xt, yt, xv, yv = _split(x, y, val_fraction, g)
    return DataBundle(xt, yt, xv, yv, "regression", (), name=f"{curve} curve",
                      notes="Regression: the network predicts a number, not a class.")


_ARITHMETIC = {
    "a + b": (lambda a, b: a + b, True),
    "a × b": (lambda a, b: a * b, False),
    "larger of a, b": (lambda a, b: torch.maximum(a, b), False),
    "a² + b²": (lambda a, b: a * a + b * b, False),
}
"""Each entry says what to compute and whether one Linear layer can do it.

The flag is the whole point of the node. ``a + b`` is a weighted sum of the inputs, which
is the literal definition of a Linear layer, so a network with no hidden layer and no
activation gets it exactly. ``a × b`` is not, and the same network cannot do better than a
plane through the middle of a saddle. Switching one combo turns a solved problem into an
unsolvable one without touching the network, which is a shorter argument for nonlinearity
than any amount of prose.
"""


def arithmetic(operation: str = "a × b", n: int = 2000, noise: float = 0.0,
               span: float = 2.0, val_fraction: float = 0.25, seed: int = 0) -> DataBundle:
    """Two numbers in, one number out. The smallest problem that still needs a hidden layer."""
    if operation not in _ARITHMETIC:
        raise NeurodesError(f"Unknown operation {operation!r}",
                            hint="Choose one of: " + ", ".join(_ARITHMETIC))
    fn, linear = _ARITHMETIC[operation]
    with allocating():
        g = torch.Generator().manual_seed(int(seed))
        x = (torch.rand(max(int(n), 8), 2, generator=g) * 2 - 1) * float(span)
        y = fn(x[:, :1], x[:, 1:2])
        if noise:
            y = y + torch.randn(y.shape, generator=g) * float(noise)
        xt, yt, xv, yv = _split(x, y, val_fraction, g)
    reach = "A Linear layer on its own can do this exactly." if linear else \
        "No Linear layer can do this, however many you stack, unless something bends."
    return DataBundle(xt, yt, xv, yv, "regression", (), name=f"{operation} on [-{span}, {span}]",
                      notes=f"Regression from two numbers to one. {reach}")


# ---------------------------------------------------------------------------
# Image datasets, if torchvision is around
# ---------------------------------------------------------------------------

_VISION = {
    "MNIST": ("MNIST", tuple(str(i) for i in range(10))),
    "FashionMNIST": ("FashionMNIST", ("t-shirt", "trouser", "pullover", "dress", "coat",
                                      "sandal", "shirt", "sneaker", "bag", "boot")),
    "CIFAR10": ("CIFAR10", ("plane", "car", "bird", "cat", "deer",
                            "dog", "frog", "horse", "ship", "truck")),
    "KMNIST": ("KMNIST", tuple(str(i) for i in range(10))),
}


def vision_dataset(which: str = "MNIST", root: str = "", limit_train: int = 6000,
                   limit_val: int = 1000, flatten: bool = False,
                   download: bool = True) -> DataBundle:
    """Load a small slice of a standard image dataset.

    The slice is deliberate. A few thousand examples is enough to watch a network learn,
    and it keeps a single run inside a coffee break on a CPU.
    """
    if which not in _VISION:
        raise NeurodesError(f"Unknown dataset {which!r}", hint="Choose one of: " + ", ".join(_VISION))
    try:
        import torchvision  # noqa: F401
        from torchvision import datasets, transforms
    except ImportError:
        raise NeurodesError(
            "torchvision is not installed, so the image datasets are unavailable.",
            hint="Install it with 'pip install torchvision', or use one of the built-in toy "
                 "datasets, which need no download.",
        ) from None

    import os
    root = root or os.path.join(os.path.expanduser("~"), ".cache", "neurodes-data")
    cls_name, classes = _VISION[which]
    ctor = getattr(datasets, cls_name)
    tf = transforms.ToTensor()
    try:
        train = ctor(root=root, train=True, download=download, transform=tf)
        val = ctor(root=root, train=False, download=download, transform=tf)
    except Exception as exc:
        raise NeurodesError(
            f"Could not load {which}: {exc}",
            hint=f"It downloads to {root} on first use. Check the network connection, or "
                 "use a toy dataset instead.",
        ) from None

    def take(ds, limit):
        limit = min(int(limit), len(ds)) if limit > 0 else len(ds)
        xs = torch.stack([ds[i][0] for i in range(limit)])
        ys = torch.tensor([int(ds[i][1]) for i in range(limit)], dtype=torch.long)
        return xs, ys

    with allocating():
        xt, yt = take(train, limit_train)
        xv, yv = take(val, limit_val)
        if flatten:
            xt, xv = xt.flatten(1), xv.flatten(1)
    return DataBundle(xt, yt, xv, yv, "classification", classes,
                      name=f"{which} ({xt.shape[0]} train / {xv.shape[0]} val)",
                      notes="Pixel values are scaled to 0-1.")


# ---------------------------------------------------------------------------
# From tensors the user already has
# ---------------------------------------------------------------------------

#: Extensions worth trying to open. Anything else in the folder is ignored silently.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")


def image_folder(path: str, size: int = 32, greyscale: bool = False,
                 val_fraction: float = 0.2, max_per_class: int = 0,
                 seed: int = 0) -> DataBundle:
    """Load a directory of class subfolders.

    The layout is the one everybody already uses, and the one torchvision's ImageFolder
    expects, so a dataset downloaded from anywhere will usually just work::

        my_data/
            cat/   img001.png  img002.png ...
            dog/   img003.png ...

    Each subfolder is a class, named after the folder. Done with Pillow rather than
    torchvision so that this works on any ComfyUI install.
    """
    import os

    from PIL import Image

    root = os.path.expanduser(str(path).strip().strip('"'))
    if not root:
        raise NeurodesError("No folder given.",
                            hint="Type the path to a folder of class subfolders.")
    if not os.path.isdir(root):
        raise NeurodesError(
            f"There is no folder at {root}.",
            hint="Check the path. It should be the folder that *contains* the class "
                 "folders, not one of the class folders itself.",
        )

    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not classes:
        loose = [f for f in os.listdir(root) if f.lower().endswith(IMAGE_SUFFIXES)]
        raise NeurodesError(
            f"{root} has no subfolders, so there are no classes to learn."
            + (f" It does contain {len(loose)} image(s) loose at the top level."
               if loose else ""),
            hint="Put the images into one folder per class, e.g. my_data/cat/... and "
                 "my_data/dog/..., then point this node at my_data.",
        )

    side = max(4, int(size))
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    counts: dict[str, int] = {}
    skipped = 0

    for label, name in enumerate(classes):
        folder = os.path.join(root, name)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_SUFFIXES))
        if max_per_class > 0:
            files = files[: int(max_per_class)]
        for filename in files:
            try:
                with Image.open(os.path.join(folder, filename)) as handle:
                    picture = handle.convert("L" if greyscale else "RGB")
                    picture = picture.resize((side, side), Image.BILINEAR)
                    array = torch.frombuffer(picture.tobytes(), dtype=torch.uint8).clone()
            except Exception:
                skipped += 1
                continue
            channels = 1 if greyscale else 3
            xs.append(array.reshape(side, side, channels).permute(2, 0, 1).float() / 255.0)
            ys.append(label)
        counts[name] = len(files)

    if not xs:
        raise NeurodesError(
            f"Found {len(classes)} class folder(s) in {root} but no readable images in them.",
            hint="Supported types are " + ", ".join(IMAGE_SUFFIXES) + ".",
        )
    empty = [n for n, c in counts.items() if c == 0]
    if empty:
        raise NeurodesError(
            "These class folders are empty: " + ", ".join(empty) + ".",
            hint="Every class needs at least one image, or remove the folder.",
        )
    if len(classes) < 2:
        raise NeurodesError(
            f"Only one class ({classes[0]}) was found, so there is nothing to tell apart.",
            hint="A classifier needs at least two class folders.",
        )

    with allocating():
        x = torch.stack(xs)
        y = torch.tensor(ys, dtype=torch.long)

    note = ", ".join(f"{n}: {c}" for n, c in counts.items())
    if skipped:
        note += f"  ({skipped} file(s) could not be read)"
    bundle = from_tensors(x, y, "classification", val_fraction, tuple(classes),
                          name=f"{os.path.basename(root) or root} ({len(xs)} images)", seed=seed)
    bundle.notes = note
    return bundle


def as_autoencoder(bundle: DataBundle, name: str = "") -> DataBundle:
    """Make a dataset its own target.

    An autoencoder is trained to reproduce its input, so the labels are thrown away and
    the target becomes the input itself. Any dataset can be turned into this, which is why
    it is a conversion rather than a separate loader.
    """
    return DataBundle(
        bundle.x_train, bundle.x_train.clone(),
        bundle.x_val, bundle.x_val.clone(),
        task="reconstruction", classes=(),
        name=name or f"{bundle.name} (reconstruction)",
        notes="Labels discarded: the target is the input itself.",
    )


def from_tensors(x: torch.Tensor, y: torch.Tensor, task: str = "classification",
                 val_fraction: float = 0.2, classes: tuple[str, ...] = (),
                 name: str = "custom", seed: int = 0) -> DataBundle:
    """Wrap tensors the caller already built, splitting off a validation set."""
    if x.shape[0] != y.shape[0]:
        raise NeurodesError(
            f"Inputs and targets disagree: {x.shape[0]} inputs but {y.shape[0]} targets.",
            hint="There has to be exactly one target per example.",
        )
    if x.shape[0] < 2:
        raise NeurodesError("Need at least two examples to make a dataset.")
    # These may have come from the host (a ComfyUI IMAGE, say), in which case they are
    # inference tensors and cannot be saved for backward. adopt() copies them out.
    x, y = adopt(x), adopt(y)
    with allocating():
        g = torch.Generator().manual_seed(int(seed))
        if task == "classification":
            y = y.long().reshape(-1)
        else:
            y = y.float()
            if y.dim() == 1:
                y = y.unsqueeze(1)
        xt, yt, xv, yv = _split(
            x.float() if task == "regression" or x.is_floating_point() else x,
            y, val_fraction, g)
    return DataBundle(xt, yt, xv, yv, task, classes, name=name)


def images_to_dataset(images: torch.Tensor, labels: torch.Tensor,
                      channels_first: bool = True, val_fraction: float = 0.2,
                      classes: tuple[str, ...] = (), name: str = "images") -> DataBundle:
    """Turn a ComfyUI IMAGE batch [B, H, W, C] into a dataset.

    ComfyUI keeps channels last; torch convolutions want them first, so this converts by
    default rather than leaving a shape trap for the user to fall into.
    """
    if images.dim() != 4:
        raise NeurodesError(
            "Expected a batch of images shaped [batch, height, width, channels], got "
            f"{tuple(images.shape)}.",
        )
    with allocating():
        x = adopt(images).float()
        if channels_first:
            x = x.permute(0, 3, 1, 2).contiguous()
    return from_tensors(x, labels, "classification", val_fraction, classes, name)


#: Everything a node can offer in a dropdown.
TOY_NAMES = tuple(TOY_SHAPES)
CURVE_NAMES = tuple(_CURVES)
VISION_NAMES = tuple(_VISION)
