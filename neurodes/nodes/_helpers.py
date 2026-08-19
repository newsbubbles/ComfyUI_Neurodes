"""Shared machinery for the node adapters."""

from __future__ import annotations

from typing import Any, Sequence

from comfy_api.latest import io, ui

from ..core.errors import NeurodesError
from ..core.registry import VARIADIC, LayerSpec, Param
from ..core.summary import badge as shape_badge
from ..core.trace import SymTensor
from .types import Tensor

_WIDGETS = {
    "int": lambda p: io.Int.Input(p.name, display_name=p.display, default=int(p.default),
                                  min=int(p.min) if p.min is not None else -(1 << 30),
                                  max=int(p.max) if p.max is not None else (1 << 30),
                                  step=int(p.step) if p.step else None,
                                  tooltip=p.doc, advanced=p.advanced or None),
    "float": lambda p: io.Float.Input(p.name, display_name=p.display, default=float(p.default),
                                      min=float(p.min) if p.min is not None else -1e30,
                                      max=float(p.max) if p.max is not None else 1e30,
                                      step=float(p.step) if p.step else None,
                                      tooltip=p.doc, advanced=p.advanced or None),
    "bool": lambda p: io.Boolean.Input(p.name, display_name=p.display, default=bool(p.default),
                                       tooltip=p.doc, advanced=p.advanced or None),
    "string": lambda p: io.String.Input(p.name, display_name=p.display, default=str(p.default),
                                        multiline=p.multiline, tooltip=p.doc,
                                        advanced=p.advanced or None),
    "ints": lambda p: io.String.Input(p.name, display_name=p.display, default=str(p.default),
                                      tooltip=p.doc + "  One number, or one per axis "
                                                      "separated by commas.",
                                      advanced=p.advanced or None),
    "combo": lambda p: io.Combo.Input(p.name, display_name=p.display, options=list(p.choices),
                                      default=str(p.default), tooltip=p.doc,
                                      advanced=p.advanced or None),
}


def widget(p: Param):
    """Turn one core :class:`Param` into a ComfyUI input."""
    try:
        return _WIDGETS[p.kind](p)
    except KeyError:
        raise NeurodesError(f"No widget mapping for param kind {p.kind!r}") from None


def widgets(params: Sequence[Param]) -> list:
    return [widget(p) for p in params]


def camel(key: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in key.split("_") if part)


def tensor_input_ids(spec: LayerSpec) -> list[str]:
    """The socket ids a layer's tensor inputs use."""
    if spec.arity == VARIADIC:
        return ["tensors"]
    return [spec.input_label(i) for i in range(spec.arity)]


def tensor_inputs(spec: LayerSpec, max_variadic: int = 16) -> list:
    """Tensor input sockets, growing automatically when the layer takes any number."""
    if spec.arity == VARIADIC:
        template = io.Autogrow.TemplatePrefix(
            Tensor.Input("tensor"), prefix="tensor", min=2, max=max_variadic)
        return [io.Autogrow.Input("tensors", template=template,
                                  tooltip="Connect two or more tensors. A new slot appears "
                                          "as you fill the last one.")]
    return [Tensor.Input(name, tooltip=f"Input tensor '{name}'.")
            for name in tensor_input_ids(spec)]


def collect_tensors(spec: LayerSpec, kwargs: dict) -> list[SymTensor]:
    """Pull the tensor arguments back out of the kwargs ComfyUI hands to ``execute``."""
    if spec.arity == VARIADIC:
        grown = kwargs.get("tensors") or {}
        return [v for v in grown.values() if v is not None]
    return [kwargs.get(name) for name in tensor_input_ids(spec)]


SHARE_INPUT = io.String.Input(
    "share", default="", advanced=True,
    tooltip="Weight sharing. Give two layers the same tag and they use one set of weights "
            "instead of two — that is how a Siamese network compares two things with the "
            "same eyes. Leave it empty for a normal, independent layer.")

LABEL_INPUT = io.String.Input(
    "label", default="", advanced=True,
    tooltip="A name for this layer, used in the summary table and the exported code.")


def badge_output(tensor: SymTensor, params: int = 0):
    """The little line of text a layer node shows after it runs."""
    return ui.PreviewText(shape_badge(tensor, params))


def save_inputs(default_name: str) -> list:
    """The two widgets every image-producing node gets.

    Previews live in ``temp/`` and are cleared; in this pack the pictures often *are* the
    deliverable, so there has to be a one-click way to keep them. Saved files land in
    ``output/neurodes/`` and carry the workflow in their PNG metadata, which means a saved
    chart can be dragged back onto the canvas to restore the graph that produced it.
    """
    return [
        io.Boolean.Input(
            "save", default=False,
            tooltip="Also write a PNG to ComfyUI's output folder, instead of only "
                    "previewing it. Saved images carry the workflow inside them, so one can "
                    "be dragged back onto the canvas to rebuild the graph that made it."),
        io.String.Input(
            "filename_prefix", default=f"neurodes/{default_name}", advanced=True,
            tooltip="Path under the output folder. A '/' makes a subfolder."),
    ]


def save_image_file(cls, images, filename_prefix: str = "neurodes/image"):
    """Write PNGs into ComfyUI's output folder and return what it saved."""
    return ui.ImageSaveHelper.save_images(
        images, filename_prefix=(filename_prefix or "neurodes/image"),
        folder_type=io.FolderType.output, cls=cls, compress_level=4)


def image_result(cls, images, *extra, save: bool = False,
                 filename_prefix: str = "neurodes/image"):
    """Return an IMAGE, and either preview it or save it depending on the toggle."""
    if save:
        return io.NodeOutput(images, *extra,
                             ui=ui.SavedImages(save_image_file(cls, images, filename_prefix)))
    return io.NodeOutput(images, *extra, ui=ui.PreviewImage(images, cls=cls))


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)
