"""Convolutions and pooling — the layers that look at a neighbourhood instead of everything.

Everything here is generated from one factory per family, so a 1d, 2d and 3d convolution
cannot disagree about how their output size is computed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import ShapeError
from ..registry import P, as_ints, layer, positive, require_float
from ..shape import Dim, Shape, conv_out, convt_out
from ._common import kwargs_src, out, same

CAT = "layers/conv"
CAT_POOL = "layers/pool"

_SPATIAL_NAMES = {1: ("length",), 2: ("height", "width"), 3: ("depth", "height", "width")}


def _parse_padding(value, kernel: tuple[int, ...], stride: tuple[int, ...],
                   dilation: tuple[int, ...], n: int, layer_name: str) -> tuple[int, ...]:
    """Read the padding widget: ``same``, ``valid``, or explicit numbers."""
    text = str(value).strip().lower()
    if text in ("valid", "none", ""):
        return (0,) * n
    if text == "same":
        if any(s != 1 for s in stride):
            raise ShapeError(
                f"{layer_name}: padding 'same' only makes sense with stride 1, but stride is "
                f"{stride if n > 1 else stride[0]}.",
                hint="Use stride 1 with 'same' padding, or write the padding as a number.",
            )
        pads = []
        for k, d in zip(kernel, dilation):
            eff = d * (k - 1) + 1
            if eff % 2 == 0:
                raise ShapeError(
                    f"{layer_name}: padding 'same' needs an odd kernel, but got {k}.",
                    hint="Use an odd kernel size such as 3, 5 or 7, or set the padding by hand.",
                )
            pads.append(eff // 2)
        return tuple(pads)
    return as_ints(value, n, "padding")


def _conv_geometry(cfg, n: int, layer_name: str):
    kernel = as_ints(cfg["kernel_size"], n, "kernel_size")
    stride = as_ints(cfg["stride"], n, "stride")
    dilation = as_ints(cfg.get("dilation", 1), n, "dilation")
    for name, vals in (("kernel_size", kernel), ("stride", stride), ("dilation", dilation)):
        for v in vals:
            positive(v, name, layer_name)
    padding = _parse_padding(cfg["padding"], kernel, stride, dilation, n, layer_name)
    return kernel, stride, padding, dilation


def _conv_params(n: int, transposed: bool = False):
    base = [
        P("out_channels", "int", 32, "How many feature maps this layer produces.", min=1, max=1 << 16),
        P("kernel_size", "ints", "3", "The size of the window. One number, or one per axis."),
        P("stride", "ints", "1", "How far the window moves each step. 2 halves the size."),
        P("padding", "string", "same",
          "'same' keeps the size, 'valid' adds nothing, or write numbers."),
    ]
    if transposed:
        base.append(P("output_padding", "ints", "0", "Extra size added to one side of the output.", advanced=True))
    else:
        base.append(P("dilation", "ints", "1", "Spread the window out, to see further for the same cost.", advanced=True))
    base += [
        P("groups", "int", 1, "Split the channels into independent groups. Set it to the "
                              "channel count for a depthwise convolution.", min=1, max=1 << 14, advanced=True),
        P("bias", "bool", True, "Add a learned offset per output channel."),
    ]
    return tuple(base)


def _make_conv(n: int):
    key, display = f"conv{n}d", f"Conv {n}D"
    torch_cls = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[n]

    def infer(ins, cfg):
        t = ins[0]
        require_float(t, display)
        if t.rank != n + 2:
            raise ShapeError(
                f"expects a rank-{n + 2} tensor shaped [batch, channels, "
                + ", ".join(_SPATIAL_NAMES[n]) + f"], but got {t.shape} (rank {t.rank}).",
                hint=f"A Conv {n}D reads {n} spatial dimension(s). "
                     + ("Try Conv 2D for images." if n != 2 else "Add a channel dimension with Unsqueeze if you have one."),
            )
        if not t.shape[1].is_concrete:
            raise ShapeError(f"needs a concrete channel count on axis 1, but {t.shape} has '{t.shape[1]}'.")
        in_ch, groups = t.shape[1].size, positive(cfg["groups"], "groups")
        out_ch = positive(cfg["out_channels"], "out_channels")
        if in_ch % groups or out_ch % groups:
            raise ShapeError(
                f"groups={groups} does not divide both the {in_ch} input channels and the "
                f"{out_ch} output channels.",
                hint="Set groups to 1, or to a number that divides both channel counts.",
            )
        kernel, stride, padding, dilation = _conv_geometry(cfg, n, display)
        dims = [t.shape[0], Dim(out_ch)]
        for i, name in enumerate(_SPATIAL_NAMES[n]):
            dims.append(conv_out(t.shape[i + 2], kernel[i], stride[i], padding[i], dilation[i], axis_name=name))
        return out(Shape(dims), t.dtype)

    def build(s, c):
        kernel, stride, padding, dilation = _conv_geometry(c, n, display)
        return torch_cls(s[0][1].size, positive(c["out_channels"], "out_channels"),
                         kernel_size=kernel, stride=stride, padding=padding,
                         dilation=dilation, groups=int(c["groups"]), bias=bool(c["bias"]))

    def emit(s, c):
        kernel, stride, padding, dilation = _conv_geometry(c, n, display)
        return "nn.Conv{}d({}, {}, {})".format(
            n, s[0][1].size, int(c["out_channels"]),
            kwargs_src(kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, groups=int(c["groups"]), bias=bool(c["bias"])))

    doc = {
        1: "Slides a learned window along a sequence. Good for audio and any 1d signal.",
        2: "Slides a learned window across an image. The workhorse of vision: it looks at a "
           "small patch at a time and reuses the same weights everywhere, so it needs far "
           "fewer parameters than a Linear on the same picture.",
        3: "Slides a learned window through a volume. For video and 3d scans.",
    }[n]
    layer(key, display, CAT, doc=doc,
          aliases=("convolution", f"cnn{n}d", "filter") + (("cnn", "image layer") if n == 2 else ()),
          params=_conv_params(n), build=build, emit_init=emit)(infer)


for _n in (1, 2, 3):
    _make_conv(_n)


# ---------------------------------------------------------------------------
# Transposed convolution (upsampling)
# ---------------------------------------------------------------------------

def _convt_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Conv Transpose 2D")
    if t.rank != 4:
        raise ShapeError(
            f"expects [batch, channels, height, width], but got {t.shape} (rank {t.rank}).",
            hint="Transposed convolution upsamples images, so it needs a 4-dimensional tensor.",
        )
    if not t.shape[1].is_concrete:
        raise ShapeError(f"needs a concrete channel count, got '{t.shape[1]}'.")
    kernel = as_ints(cfg["kernel_size"], 2, "kernel_size")
    stride = as_ints(cfg["stride"], 2, "stride")
    outpad = as_ints(cfg["output_padding"], 2, "output_padding")
    padding = _convt_pad(cfg)
    dims = [t.shape[0], Dim(positive(cfg["out_channels"], "out_channels"))]
    for i in range(2):
        dims.append(convt_out(t.shape[i + 2], kernel[i], stride[i], padding[i], outpad[i], 1))
    return out(Shape(dims), t.dtype)


def _convt_pad(c):
    kernel = as_ints(c["kernel_size"], 2, "kernel_size")
    text = str(c["padding"]).strip().lower()
    if text == "same":
        return tuple(k // 2 for k in kernel)
    if text == "valid":
        return (0, 0)
    return as_ints(c["padding"], 2, "padding")


layer(
    "conv_transpose2d", "Conv Transpose 2D", CAT,
    doc="A convolution run backwards: it makes the image bigger instead of smaller. This is "
        "the standard way a decoder or a generator grows a small feature map back into a picture.",
    aliases=("deconvolution", "upsample conv", "transposed convolution", "generator"),
    params=_conv_params(2, transposed=True),
    build=lambda s, c: nn.ConvTranspose2d(
        s[0][1].size, positive(c["out_channels"], "out_channels"),
        kernel_size=as_ints(c["kernel_size"], 2, "kernel_size"),
        stride=as_ints(c["stride"], 2, "stride"), padding=_convt_pad(c),
        output_padding=as_ints(c["output_padding"], 2, "output_padding"),
        groups=int(c["groups"]), bias=bool(c["bias"])),
    emit_init=lambda s, c: "nn.ConvTranspose2d({}, {}, {})".format(
        s[0][1].size, int(c["out_channels"]),
        kwargs_src(kernel_size=as_ints(c["kernel_size"], 2, "kernel_size"),
                   stride=as_ints(c["stride"], 2, "stride"), padding=_convt_pad(c),
                   output_padding=as_ints(c["output_padding"], 2, "output_padding"),
                   groups=int(c["groups"]), bias=bool(c["bias"]))),
)(_convt_infer)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _make_pool(n: int, mode: str):
    key = f"{mode}pool{n}d"
    display = f"{'Max' if mode == 'max' else 'Average'} Pool {n}D"
    cls = {("max", 1): nn.MaxPool1d, ("max", 2): nn.MaxPool2d,
           ("avg", 1): nn.AvgPool1d, ("avg", 2): nn.AvgPool2d}[(mode, n)]

    def geometry(cfg):
        kernel = as_ints(cfg["kernel_size"], n, "kernel_size")
        stride_raw = str(cfg["stride"]).strip().lower()
        stride = kernel if stride_raw in ("", "same as kernel", "kernel") else as_ints(cfg["stride"], n, "stride")
        padding = as_ints(cfg["padding"], n, "padding")
        return kernel, stride, padding

    def infer(ins, cfg):
        t = ins[0]
        require_float(t, display)
        if t.rank != n + 2:
            raise ShapeError(
                f"expects a rank-{n + 2} tensor [batch, channels, "
                + ", ".join(_SPATIAL_NAMES[n]) + f"], but got {t.shape}.",
                hint=f"Pooling keeps the channel count and shrinks the {n} spatial dimension(s).",
            )
        kernel, stride, padding = geometry(cfg)
        dims = [t.shape[0], t.shape[1]]
        for i, name in enumerate(_SPATIAL_NAMES[n]):
            dims.append(conv_out(t.shape[i + 2], kernel[i], stride[i], padding[i], 1, axis_name=name))
        return out(Shape(dims), t.dtype)

    doc = ("Takes the largest value in each window. Shrinks the picture while keeping the "
           "strongest response, which makes the network care less about exactly where a "
           "feature was." if mode == "max" else
           "Takes the mean of each window. A gentler way to shrink than max pooling.")

    layer(key, display, CAT_POOL, doc=doc,
          aliases=("downsample", "pooling", "shrink"),
          params=(
              P("kernel_size", "ints", "2", "The window size."),
              P("stride", "ints", "", "How far the window moves. Leave empty to match the kernel."),
              P("padding", "ints", "0", "Zeros added around the edge.", advanced=True),
          ),
          build=lambda s, c, _cls=cls, _g=geometry: _cls(*_g(c)),
          emit_init=lambda s, c, _n=n, _mode=mode, _g=geometry: "nn.{}Pool{}d({})".format(
              "Max" if _mode == "max" else "Avg", _n,
              kwargs_src(**dict(zip(("kernel_size", "stride", "padding"), _g(c))))),
          trains=False)(infer)


for _n in (1, 2):
    for _mode in ("max", "avg"):
        _make_pool(_n, _mode)


def _adaptive_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Adaptive Avg Pool 2D")
    if t.rank != 4:
        raise ShapeError(f"expects [batch, channels, height, width], but got {t.shape}.")
    size = as_ints(cfg["output_size"], 2, "output_size")
    for v in size:
        positive(v, "output_size", "Adaptive Avg Pool 2D")
    return out(Shape([t.shape[0], t.shape[1], Dim(size[0]), Dim(size[1])]), t.dtype)


layer(
    "adaptive_avgpool2d", "Adaptive Avg Pool 2D", CAT_POOL,
    doc="Pools to a size you choose, whatever came in. This is how a convolutional network "
        "accepts images of any size and still hands a fixed-length vector to its classifier.",
    aliases=("adaptive pooling", "any size", "resize features"),
    params=(P("output_size", "ints", "1", "The height and width you want out."),),
    build=lambda s, c: nn.AdaptiveAvgPool2d(as_ints(c["output_size"], 2, "output_size")),
    emit_init=lambda s, c: f"nn.AdaptiveAvgPool2d({as_ints(c['output_size'], 2, 'output_size')})",
    trains=False,
)(_adaptive_infer)


def _global_pool_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Global Pool")
    if t.rank < 3:
        raise ShapeError(
            f"expects at least [batch, channels, something], but got {t.shape}.",
            hint="Global pooling collapses every spatial dimension, so there has to be one.",
        )
    return out(Shape([t.shape[0], t.shape[1]]), t.dtype)


layer(
    "global_pool", "Global Pool", CAT_POOL,
    doc="Collapses every spatial dimension into a single number per channel, turning a stack "
        "of feature maps into one vector. The usual bridge from a convolutional trunk to a "
        "classifier, and much lighter than flattening.",
    aliases=("global average pooling", "gap", "squeeze spatial"),
    params=(P("mode", "combo", "mean", "Average or maximum over the spatial dimensions.",
              choices=("mean", "max")),),
    build=None,
    apply=lambda m, ts, c: (
        ts[0].mean(dim=tuple(range(2, ts[0].dim())))
        if c["mode"] == "mean" else
        ts[0].amax(dim=tuple(range(2, ts[0].dim())))
    ),
    emit_call=lambda attr, args, c: (
        f"{args[0]}.mean(dim=tuple(range(2, {args[0]}.dim())))" if c["mode"] == "mean"
        else f"{args[0]}.amax(dim=tuple(range(2, {args[0]}.dim())))"),
    trains=False,
)(_global_pool_infer)


def _upsample_infer(ins, cfg):
    t = ins[0]
    require_float(t, "Upsample")
    if t.rank != 4:
        raise ShapeError(f"expects [batch, channels, height, width], but got {t.shape}.")
    factor = positive(int(cfg["scale_factor"]), "scale_factor", "Upsample")
    dims = [t.shape[0], t.shape[1]]
    for i in (2, 3):
        d = t.shape[i]
        dims.append(Dim(d.size * factor) if d.is_concrete else Dim(f"{d}_up"))
    return out(Shape(dims), t.dtype)


layer(
    "upsample", "Upsample", CAT_POOL,
    doc="Makes the image bigger by copying or blending existing values. Unlike a transposed "
        "convolution it learns nothing, which makes it cheap and free of checkerboard artefacts.",
    aliases=("resize", "interpolate", "scale up"),
    params=(
        P("scale_factor", "int", 2, "How many times bigger.", min=2, max=16),
        P("mode", "combo", "nearest", "How new pixels are filled in.",
          choices=("nearest", "bilinear", "bicubic")),
    ),
    build=None,
    apply=lambda m, ts, c: F.interpolate(
        ts[0], scale_factor=int(c["scale_factor"]), mode=str(c["mode"]),
        align_corners=False if c["mode"] in ("bilinear", "bicubic") else None),
    emit_call=lambda attr, args, c: "F.interpolate({}, scale_factor={}, mode={!r}{})".format(
        args[0], int(c["scale_factor"]), str(c["mode"]),
        ", align_corners=False" if c["mode"] in ("bilinear", "bicubic") else ""),
    trains=False,
)(_upsample_infer)
