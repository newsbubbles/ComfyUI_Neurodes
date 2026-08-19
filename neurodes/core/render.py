"""Turning raw tensors into ComfyUI images.

This is deliberately not :mod:`neurodes.core.plot`. Those draw *charts* — axes, labels,
legends — which is right for understanding something and useless as input to anything else.
These draw the tensor and nothing but the tensor: no chrome, native resolution, values
mapped straight to pixels.

The important choice is the batch convention. A convolution activation is
``[batch, channels, height, width]``, and the interesting axis is channels. Rendered as a
contact sheet you get one image to look at; rendered as a **batch** you get one image per
channel, and every node in ComfyUI already knows what to do with a batch of images. That
one decision is what connects this to video nodes, upscalers, ControlNet and img2img
without any of them needing to know what an activation is.

Colormaps are hand-rolled from control points. matplotlib would be one import and one more
way for the pack to fail to load.
"""

from __future__ import annotations

import math

import torch

#: Control points at 0, 1/8, ... 1. Interpolated linearly, which is close enough that the
#: difference from the real thing is invisible at 8 bits.
_MAPS: dict[str, list[tuple[int, int, int]]] = {
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
                (31, 158, 137), (53, 183, 121), (109, 205, 89), (253, 231, 37)],
    "magma": [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122),
              (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191)],
    "inferno": [(0, 0, 4), (31, 12, 72), (85, 15, 109), (136, 34, 106), (186, 54, 85),
                (227, 89, 51), (249, 142, 9), (249, 201, 50), (252, 255, 164)],
    "plasma": [(13, 8, 135), (75, 3, 161), (125, 3, 168), (168, 34, 150), (203, 70, 121),
               (229, 107, 93), (248, 148, 65), (253, 195, 40), (240, 249, 33)],
    "turbo": [(48, 18, 59), (70, 107, 227), (54, 175, 238), (48, 227, 182), (110, 254, 101),
              (188, 246, 50), (243, 198, 45), (254, 123, 31), (165, 17, 2)],
    "ice": [(3, 5, 26), (10, 30, 66), (14, 62, 110), (18, 98, 148), (40, 137, 172),
            (86, 174, 190), (144, 205, 210), (200, 230, 234), (245, 252, 255)],
    "ember": [(4, 2, 2), (48, 8, 6), (99, 16, 10), (148, 34, 12), (191, 66, 16),
              (222, 111, 30), (240, 163, 62), (250, 212, 130), (255, 249, 214)],
    "grey": [(0, 0, 0), (32, 32, 32), (64, 64, 64), (96, 96, 96), (128, 128, 128),
             (160, 160, 160), (192, 192, 192), (224, 224, 224), (255, 255, 255)],
}

#: Diverging, for anything with a meaningful zero: weights, gradients, differences.
_SIGNED = {
    "cold-hot": [(94, 168, 255), (24, 26, 32), (255, 138, 96)],
    "green-magenta": [(120, 214, 148), (22, 24, 28), (222, 130, 220)],
}

COLORMAPS = tuple(_MAPS) + tuple(_SIGNED)
NORMALIZERS = ("per image", "whole tensor", "signed", "none")


def _lut(name: str, steps: int = 256) -> torch.Tensor:
    """Build a ``[steps, 3]`` lookup table in 0..1 by interpolating the control points."""
    points = _MAPS.get(name) or _SIGNED.get(name)
    if points is None:
        raise KeyError(name)
    anchors = torch.tensor(points, dtype=torch.float32) / 255.0
    position = torch.linspace(0, len(anchors) - 1, steps)
    low = position.floor().long().clamp(max=len(anchors) - 1)
    high = position.ceil().long().clamp(max=len(anchors) - 1)
    frac = (position - low.float()).unsqueeze(1)
    return anchors[low] * (1 - frac) + anchors[high] * frac


_LUT_CACHE: dict[str, torch.Tensor] = {}


def colormap(name: str) -> torch.Tensor:
    if name not in _LUT_CACHE:
        _LUT_CACHE[name] = _lut(name)
    return _LUT_CACHE[name]


def normalize(planes: torch.Tensor, mode: str = "per image") -> torch.Tensor:
    """Map values into 0..1.

    ``per image`` rescales each plane on its own, which makes faint channels legible but
    hides how strongly they actually fired. ``whole tensor`` uses one range for all of
    them, which is the honest comparison. ``signed`` centres zero at the middle of the
    range, which is what you want for weights and gradients.
    """
    flat = planes.reshape(planes.shape[0], -1)
    if mode == "none":
        return planes.clamp(0, 1)
    if mode == "signed":
        span = flat.abs().max().clamp(min=1e-8)
        return (planes / span) * 0.5 + 0.5
    if mode == "whole tensor":
        lo, hi = flat.min(), flat.max()
        return (planes - lo) / (hi - lo).clamp(min=1e-8)
    lo = flat.min(dim=1).values.reshape(-1, 1, 1)
    hi = flat.max(dim=1).values.reshape(-1, 1, 1)
    return (planes - lo) / (hi - lo).clamp(min=1e-8)


def _planes(tensor: torch.Tensor, example: int = 0) -> torch.Tensor:
    """Reduce whatever came in to ``[n, height, width]``.

    Handles the four shapes that actually turn up: a conv activation, a sequence of feature
    vectors, a batch of flat vectors, and a bare vector.
    """
    t = tensor.detach().float().cpu()
    if t.dim() >= 4:                      # [B, C, H, W, ...] -> that example's channels
        t = t[min(example, t.shape[0] - 1)]
        while t.dim() > 3:
            t = t[0]
        return t
    if t.dim() == 3:                      # [B, T, F] -> that example as one T x F picture
        return t[min(example, t.shape[0] - 1)].unsqueeze(0)
    if t.dim() == 2:                      # [B, F] -> the whole batch as one picture
        return t.unsqueeze(0)
    return t.reshape(1, 1, -1)


def _upscale(planes: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return planes
    return planes.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)


def _tile(planes: torch.Tensor, gap: int = 1, columns: int = 0) -> torch.Tensor:
    """Lay planes out in a grid, separated by a gap of the lowest value."""
    n, h, w = planes.shape
    cols = columns if columns > 0 else max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    sheet = torch.zeros(rows * (h + gap) + gap, cols * (w + gap) + gap)
    for i in range(n):
        r, c = divmod(i, cols)
        sheet[gap + r * (h + gap): gap + r * (h + gap) + h,
              gap + c * (w + gap): gap + c * (w + gap) + w] = planes[i]
    return sheet.unsqueeze(0)


def to_images(tensor: torch.Tensor, *, layout: str = "batch", colormap_name: str = "viridis",
              normalization: str = "per image", upscale: int = 1, example: int = 0,
              channel: int = -1, columns: int = 0, gap: int = 1) -> torch.Tensor:
    """Render a tensor as a ComfyUI IMAGE batch, ``[n, height, width, 3]`` in 0..1.

    ``layout="batch"`` gives one image per channel, which is what makes the result usable
    by the rest of ComfyUI. ``layout="sheet"`` gives a single tiled contact sheet, which is
    what you want when you are looking rather than piping.
    """
    planes = _planes(tensor, example)
    if channel >= 0:
        if channel >= planes.shape[0]:
            raise IndexError(
                f"channel {channel} does not exist; this tensor has {planes.shape[0]}"
            )
        planes = planes[channel: channel + 1]
    planes = normalize(planes, normalization)
    if layout == "sheet":
        planes = _tile(planes, gap=gap, columns=columns)
    planes = _upscale(planes.clamp(0, 1), max(1, int(upscale)))

    lut = colormap(colormap_name)
    index = (planes * (lut.shape[0] - 1)).round().long().clamp(0, lut.shape[0] - 1)
    return lut[index]          # [n, h, w] -> [n, h, w, 3]


def image_to_model_input(images: torch.Tensor, shape, greyscale: bool = False,
                         resize: bool = True) -> torch.Tensor:
    """ComfyUI IMAGE ``[B, H, W, C]`` into whatever the model's Input node declared.

    Handles the two conversions that always bite: ComfyUI puts channels last and torch
    wants them first, and a model trained on one-channel images cannot read three.
    """
    import torch.nn.functional as F

    x = images.detach().float()
    if x.dim() != 4:
        raise ValueError(f"expected a batch of images, got {tuple(x.shape)}")
    want_channels = shape[1].size if shape.rank == 4 and shape[1].is_concrete else x.shape[-1]
    if greyscale or want_channels == 1:
        x = x.mean(dim=-1, keepdim=True)
    elif want_channels == 3 and x.shape[-1] == 1:
        x = x.repeat(1, 1, 1, 3)
    x = x.permute(0, 3, 1, 2).contiguous()
    if resize and shape.rank == 4 and shape[2].is_concrete and shape[3].is_concrete:
        target = (shape[2].size, shape[3].size)
        if x.shape[-2:] != target:
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
    return x


def model_input_to_image(x: torch.Tensor) -> torch.Tensor:
    """The other direction: ``[B, C, H, W]`` back to a ComfyUI IMAGE."""
    t = x.detach().float().cpu()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = t.permute(0, 2, 3, 1)
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    elif t.shape[-1] > 3:
        t = t[..., :3]
    return t.clamp(0, 1)


def tile_batch(images: torch.Tensor, columns: int = 0, gap: int = 2) -> torch.Tensor:
    """A batch of ComfyUI images ``[N, H, W, C]`` laid out as one contact sheet.

    Used to turn each step of a sampling run into a single frame, so a whole batch
    evolving can go straight into a video node as an ordinary image sequence.
    """
    n, h, w, c = images.shape
    if n == 0:
        return images
    cols = int(columns) if columns else max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    sheet = images.new_zeros((rows * (h + gap) - gap, cols * (w + gap) - gap, c))
    for i in range(n):
        r, k = divmod(i, cols)
        sheet[r * (h + gap):r * (h + gap) + h, k * (w + gap):k * (w + gap) + w] = images[i]
    return sheet


def describe(tensor: torch.Tensor) -> str:
    """One line of honest statistics, for the node to show."""
    t = tensor.detach().float()
    dead = (t.abs() < 1e-6).float().mean().item() * 100
    return (f"{tuple(tensor.shape)}   min {t.min():.3f}  max {t.max():.3f}  "
            f"mean {t.mean():.3f}  sd {t.std():.3f}   {dead:.0f}% at zero")
