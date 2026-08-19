"""Preparing image data.

Two things stand between "a folder of pictures" and "something you can train on", and both
are here.

**Augmentation.** Small datasets overfit, and the first honest thing that happens when
someone trains on their own two hundred photographs is that the validation loss turns back
up. Flipping and jittering the training images is the cheapest fix there is: it costs no new
data and teaches the network that a cat is still a cat three degrees to the left.

**Pairing.** Classification needs a label per image. Image-to-image needs a *second image*
per image, and most people do not have one. Degrading a copy manufactures the pair: add
noise and the target is the clean original, and now you have a denoiser to train. That is
how you get a U-Net out of one folder of holiday photos.

Everything is torch and Pillow, so nothing new is required.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import DataBundle
from .errors import NeurodesError
from .runtime import allocating

IMAGE_TASKS = ("denoise", "blur", "colourise", "super resolution", "inpaint", "none")


def _require_images(x: torch.Tensor, what: str) -> None:
    if x.dim() != 4:
        raise NeurodesError(
            f"{what} needs image data shaped [batch, channels, height, width], but this "
            f"dataset is {tuple(x.shape)}.",
            hint="Use one of the image dataset nodes, and leave 'flatten' off.",
        )


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _affine_batch(x: torch.Tensor, rotate: float, zoom: float, shift: float,
                  generator) -> torch.Tensor:
    """Rotate, scale and translate each image by its own random amount.

    One ``grid_sample`` does all three at once, which is both faster and less code than
    three separate passes, and avoids resampling the picture three times.
    """
    n = x.shape[0]
    angles = (torch.rand(n, generator=generator) * 2 - 1) * math.radians(rotate)
    scales = 1.0 + (torch.rand(n, generator=generator) * 2 - 1) * zoom
    tx = (torch.rand(n, generator=generator) * 2 - 1) * shift
    ty = (torch.rand(n, generator=generator) * 2 - 1) * shift

    cos, sin = torch.cos(angles) / scales, torch.sin(angles) / scales
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty

    grid = F.affine_grid(theta.to(x.dtype), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid.to(x.device), mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def augment(bundle: DataBundle, copies: int = 2, flip_horizontal: bool = True,
            flip_vertical: bool = False, rotate: float = 12.0, zoom: float = 0.12,
            shift: float = 0.08, brightness: float = 0.15, noise: float = 0.0,
            seed: int = 0) -> DataBundle:
    """Grow the training split with jittered copies. The validation split is left alone.

    Augmenting validation data would be a mistake: it is meant to be a fixed yardstick, and
    a yardstick that changes every run measures nothing.
    """
    _require_images(bundle.x_train, "Augment")
    copies = max(0, int(copies))
    if copies == 0:
        return bundle

    with allocating():
        generator = torch.Generator().manual_seed(int(seed))
        batches = [bundle.x_train]
        targets = [bundle.y_train]
        target_is_image = bundle.y_train.dim() == 4

        for _ in range(copies):
            x = bundle.x_train.clone()
            y = bundle.y_train.clone()
            n = x.shape[0]

            if flip_horizontal:
                pick = torch.rand(n, generator=generator) < 0.5
                x[pick] = torch.flip(x[pick], dims=[-1])
                if target_is_image:
                    y[pick] = torch.flip(y[pick], dims=[-1])
            if flip_vertical:
                pick = torch.rand(n, generator=generator) < 0.5
                x[pick] = torch.flip(x[pick], dims=[-2])
                if target_is_image:
                    y[pick] = torch.flip(y[pick], dims=[-2])

            if rotate > 0 or zoom > 0 or shift > 0:
                if target_is_image:
                    # The pair has to move together, or the network is asked to predict a
                    # target that no longer lines up with its input.
                    merged = torch.cat([x, y], dim=1)
                    merged = _affine_batch(merged, rotate, zoom, shift, generator)
                    x, y = merged[:, : x.shape[1]], merged[:, x.shape[1]:]
                else:
                    x = _affine_batch(x, rotate, zoom, shift, generator)

            if brightness > 0:
                factor = 1.0 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * brightness
                x = (x * factor).clamp(0, 1)
            if noise > 0:
                x = (x + torch.randn(x.shape, generator=generator) * noise).clamp(0, 1)

            batches.append(x)
            targets.append(y)

        x_train = torch.cat(batches)
        y_train = torch.cat(targets)

    return DataBundle(
        x_train, y_train, bundle.x_val, bundle.y_val, bundle.task, bundle.classes,
        name=f"{bundle.name} x{copies + 1}",
        notes=f"Training split augmented to {x_train.shape[0]} examples; "
              "validation left untouched.",
    )


# ---------------------------------------------------------------------------
# Manufacturing image-to-image pairs
# ---------------------------------------------------------------------------

def _blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 1:
        return x
    size = radius * 2 + 1
    channels = x.shape[1]
    coords = torch.arange(size, dtype=torch.float32) - radius
    kernel1d = torch.exp(-(coords ** 2) / (2 * (radius / 2 + 1e-6) ** 2))
    kernel1d = (kernel1d / kernel1d.sum()).to(x)
    horizontal = kernel1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)


def _degrade(clean: torch.Tensor, task: str, strength: float, generator) -> torch.Tensor:
    if task == "denoise":
        return (clean + torch.randn(clean.shape, generator=generator) * strength).clamp(0, 1)
    if task == "blur":
        return _blur(clean, max(1, int(round(strength * 8))))
    if task == "colourise":
        # Rec. 601 luma if it is a colour image, otherwise it is already grey.
        if clean.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(clean)
            return (clean * weights).sum(dim=1, keepdim=True)
        return clean
    if task == "super resolution":
        factor = max(2, int(round(2 + strength * 6)))
        h, w = clean.shape[-2], clean.shape[-1]
        small = F.interpolate(clean, size=(max(1, h // factor), max(1, w // factor)),
                              mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="nearest")
    if task == "inpaint":
        damaged = clean.clone()
        n, _, h, w = clean.shape
        holes = max(1, int(round(strength * 8)))
        side_h, side_w = max(2, h // 6), max(2, w // 6)
        for i in range(n):
            for _ in range(holes):
                top = int(torch.randint(0, max(1, h - side_h), (1,), generator=generator))
                left = int(torch.randint(0, max(1, w - side_w), (1,), generator=generator))
                damaged[i, :, top:top + side_h, left:left + side_w] = 0.0
        return damaged
    if task == "none":
        return clean
    raise NeurodesError(f"Unknown image task {task!r}",
                        hint="Choose one of: " + ", ".join(IMAGE_TASKS))


def as_image_task(bundle: DataBundle, task: str = "denoise", strength: float = 0.25,
                  seed: int = 0) -> DataBundle:
    """Turn a folder of pictures into an image-to-image problem.

    The clean image becomes the target and a spoiled copy becomes the input, so a dataset
    that had only labels — or no labels at all — can train a U-Net. Nothing is annotated by
    hand, which is the point: the supervision is manufactured from the damage.
    """
    _require_images(bundle.x_train, "Image task")
    if task not in IMAGE_TASKS:
        raise NeurodesError(f"Unknown image task {task!r}",
                            hint="Choose one of: " + ", ".join(IMAGE_TASKS))

    with allocating():
        generator = torch.Generator().manual_seed(int(seed))
        x_train = _degrade(bundle.x_train, task, float(strength), generator)
        x_val = _degrade(bundle.x_val, task, float(strength), generator)
        y_train, y_val = bundle.x_train.clone(), bundle.x_val.clone()

    return DataBundle(
        x_train, y_train, x_val, y_val, task="reconstruction", classes=(),
        name=f"{bundle.name} ({task})",
        notes=f"Input is a {task} of the target. The clean image is the answer, so no "
              "labelling was needed.",
    )


def as_pairs(bundle: DataBundle, pairs: int = 4, seed: int = 0) -> DataBundle:
    """Turn a labelled dataset into "are these two the same thing?".

    Two examples go in and one bit comes out, which is the problem a Siamese network exists
    for: it never learns the classes, it learns a *comparison*, so it can be asked about
    classes it has never seen. Half the pairs are drawn from the same class and half from
    different ones, so chance is 50% and the number means something.

    The two examples arrive as two separate inputs, which is why this needs a model with two
    Input nodes — and one set of weights shared between them, or the two towers see the
    world differently and the comparison is meaningless.
    """
    if bundle.task != "classification":
        raise NeurodesError(
            f"Pairs need labels to know what 'the same' means, but {bundle.name} is a "
            f"{bundle.task} dataset.",
            hint="Use a classification dataset — a Toy Dataset, an Image Dataset, or an "
                 "Image Folder Dataset with one subfolder per class.")
    if bundle.side_train:
        raise NeurodesError("This dataset already has more than one input.",
                            hint="Pairs are built from a plain single-input dataset.")

    with allocating():
        g = torch.Generator().manual_seed(int(seed))

        def build(x: torch.Tensor, y: torch.Tensor, repeats: int):
            n = int(x.shape[0])
            if n < 2:
                return x.clone(), x.clone(), y.new_zeros((n,))
            labels = y.reshape(-1).long()
            by_class = {int(c): (labels == int(c)).nonzero().reshape(-1)
                        for c in labels.unique()}
            usable = [c for c, idx in by_class.items() if idx.numel() >= 2]
            left, right, same = [], [], []
            for _ in range(max(1, int(repeats))):
                for i in range(n):
                    c = int(labels[i])
                    want_same = bool(torch.randint(0, 2, (1,), generator=g)) and c in usable
                    if want_same:
                        pool = by_class[c]
                    else:
                        others = [k for k in by_class if k != c]
                        if not others:
                            continue
                        pick = others[int(torch.randint(0, len(others), (1,), generator=g))]
                        pool = by_class[pick]
                    j = int(pool[int(torch.randint(0, pool.numel(), (1,), generator=g))])
                    if j == i and pool.numel() > 1:
                        j = int(pool[(int((pool == i).nonzero()[0]) + 1) % pool.numel()])
                    left.append(i)
                    right.append(j)
                    same.append(1 if int(labels[j]) == c else 0)
            li = torch.tensor(left, dtype=torch.long)
            ri = torch.tensor(right, dtype=torch.long)
            return x[li], x[ri], torch.tensor(same, dtype=torch.long)

        xa, xb, ya = build(bundle.x_train, bundle.y_train, pairs)
        va, vb, yv = build(bundle.x_val, bundle.y_val, 1)

    return DataBundle(
        xa, ya, va, yv, task="classification", classes=("different", "same"),
        name=f"{bundle.name} (pairs)",
        notes="Two examples in, one bit out. Half the pairs share a class, so 50% is chance. "
              "The model needs two Input nodes and one shared tower.",
        side_train=(xb,), side_val=(vb,),
    )


def pairs_from_folders(input_folder: str, target_folder: str, size: int = 64,
                       greyscale: bool = False, target_greyscale: bool = False,
                       val_fraction: float = 0.2, limit: int = 0,
                       seed: int = 0) -> DataBundle:
    """Two parallel folders, matched by filename.

    For data that is genuinely paired — photographs and their segmentation masks, before
    and after, sketch and render. Files are matched on name without the extension, so
    ``in/042.jpg`` pairs with ``out/042.png``.
    """
    import os

    from PIL import Image

    from .data import IMAGE_SUFFIXES

    def listing(folder: str, label: str) -> dict[str, str]:
        folder = os.path.expanduser(str(folder).strip().strip('"'))
        if not os.path.isdir(folder):
            raise NeurodesError(f"There is no {label} folder at {folder}.",
                                hint="Check the path.")
        found = {}
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(IMAGE_SUFFIXES):
                found[os.path.splitext(name)[0]] = os.path.join(folder, name)
        return found

    ins, outs = listing(input_folder, "input"), listing(target_folder, "target")
    shared = sorted(set(ins) & set(outs))
    if not shared:
        raise NeurodesError(
            f"No filenames appear in both folders ({len(ins)} input, {len(outs)} target).",
            hint="Pairs are matched on the filename without its extension, so in/042.jpg "
                 "pairs with out/042.png. Rename them to match.",
        )
    if limit > 0:
        shared = shared[: int(limit)]

    side = max(4, int(size))

    def load(path: str, grey: bool) -> torch.Tensor:
        with Image.open(path) as handle:
            picture = handle.convert("L" if grey else "RGB").resize((side, side),
                                                                    Image.BILINEAR)
            raw = torch.frombuffer(picture.tobytes(), dtype=torch.uint8).clone()
        channels = 1 if grey else 3
        return raw.reshape(side, side, channels).permute(2, 0, 1).float() / 255.0

    with allocating():
        xs = torch.stack([load(ins[k], greyscale) for k in shared])
        ys = torch.stack([load(outs[k], target_greyscale) for k in shared])
        n = xs.shape[0]
        order = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
        cut = max(1, int(n * (1.0 - val_fraction)))
        train, val = order[:cut], order[cut:]

    dropped = len(set(ins) ^ set(outs))
    return DataBundle(
        xs[train], ys[train], xs[val], ys[val], task="reconstruction", classes=(),
        name=f"{len(shared)} pairs",
        notes=f"Matched on filename." + (f" {dropped} file(s) had no partner." if dropped else ""),
    )
