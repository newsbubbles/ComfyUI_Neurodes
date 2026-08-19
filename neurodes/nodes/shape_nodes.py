"""Shapes, and the Input node that turns one into a tensor.

A shape is written as text — ``B, 3, 224, 224`` — because that handles any rank in one
widget and because names like ``B`` for the batch size are how people already talk about
shapes. The preset nodes exist for anyone who would rather turn dials, and Shape From Dims
exists for shapes that have to be worked out upstream.
"""

from __future__ import annotations

from comfy_api.latest import io, ui

from ..core.errors import ParseError
from ..core.shape import Dim, Shape
from ..core.trace import DTYPES, make_input
from ._helpers import badge_output
from .types import ShapeType, Tensor, category

CAT = category("shape")

_SHAPE_HELP = (
    "Dimensions separated by commas. A number is a fixed size; a name like B or T stands "
    "for a size that is only known when the data arrives. The batch dimension is written "
    "out on purpose — hiding it is where most shape confusion starts."
)


def to_shape(value) -> Shape:
    """Accept either a parsed Shape from a socket, or the text from the widget."""
    if isinstance(value, Shape):
        return value
    if value is None:
        raise ParseError("No shape given.", hint="Type a shape like 'B, 3, 224, 224'.")
    return Shape.parse(str(value))


class NeuroShape(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroShape",
            display_name="Shape",
            category=CAT,
            description="Describes the size of a tensor. " + _SHAPE_HELP,
            search_aliases=["tensor shape", "dimensions", "size", "empty tensor"],
            inputs=[io.String.Input("dims", default="B, 1, 28, 28", tooltip=_SHAPE_HELP)],
            outputs=[ShapeType.Output(display_name="shape")],
        )

    @classmethod
    def execute(cls, dims: str) -> io.NodeOutput:
        shape = Shape.parse(dims)
        return io.NodeOutput(shape, ui=ui.PreviewText(f"{shape}  (rank {shape.rank})"))


class NeuroShapeImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroShapeImage",
            display_name="Shape (Image)",
            category=CAT,
            description="An image batch, in the [batch, channels, height, width] order that "
                        "torch convolutions expect. Note that ComfyUI's own IMAGE type puts "
                        "channels last — the dataset nodes convert for you.",
            search_aliases=["image shape", "nchw", "picture size"],
            inputs=[
                io.String.Input("batch", default="B", tooltip="A number, or a name like B."),
                io.Int.Input("channels", default=3, min=1, max=1 << 16,
                             tooltip="3 for colour, 1 for greyscale."),
                io.Int.Input("height", default=64, min=1, max=1 << 16),
                io.Int.Input("width", default=64, min=1, max=1 << 16),
            ],
            outputs=[ShapeType.Output(display_name="shape")],
        )

    @classmethod
    def execute(cls, batch: str, channels: int, height: int, width: int) -> io.NodeOutput:
        shape = Shape([_dim(batch), channels, height, width])
        return io.NodeOutput(shape, ui=ui.PreviewText(str(shape)))


class NeuroShapeSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroShapeSequence",
            display_name="Shape (Sequence)",
            category=CAT,
            description="A batch of sequences: [batch, time, features]. Leave 'time' as T if "
                        "the length varies, which it usually does.",
            search_aliases=["sequence shape", "text shape", "time series", "tokens"],
            inputs=[
                io.String.Input("batch", default="B", tooltip="A number, or a name like B."),
                io.String.Input("time", default="T", tooltip="Sequence length. A number, or a name like T."),
                io.Int.Input("features", default=128, min=1, max=1 << 20,
                             tooltip="How many values describe each step."),
            ],
            outputs=[ShapeType.Output(display_name="shape")],
        )

    @classmethod
    def execute(cls, batch: str, time: str, features: int) -> io.NodeOutput:
        shape = Shape([_dim(batch), _dim(time), features])
        return io.NodeOutput(shape, ui=ui.PreviewText(str(shape)))


class NeuroShapeVector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroShapeVector",
            display_name="Shape (Vector)",
            category=CAT,
            description="A batch of flat vectors: [batch, features]. The shape for tabular "
                        "data, and for anything that has already been flattened.",
            search_aliases=["vector shape", "flat", "tabular", "features"],
            inputs=[
                io.String.Input("batch", default="B", tooltip="A number, or a name like B."),
                io.Int.Input("features", default=2, min=1, max=1 << 22),
            ],
            outputs=[ShapeType.Output(display_name="shape")],
        )

    @classmethod
    def execute(cls, batch: str, features: int) -> io.NodeOutput:
        shape = Shape([_dim(batch), features])
        return io.NodeOutput(shape, ui=ui.PreviewText(str(shape)))


class NeuroShapeFromDims(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplatePrefix(io.Int.Input("dim", default=1), prefix="dim",
                                              min=1, max=8)
        return io.Schema(
            node_id="NeuroShapeFromDims",
            display_name="Shape From Dims",
            category=CAT,
            description="Builds a shape out of numbers coming from elsewhere in the graph. "
                        "A new slot appears each time you fill the last one. Use this when a "
                        "size is computed rather than typed; otherwise the Shape node is simpler.",
            search_aliases=["build shape", "dims", "dynamic shape"],
            inputs=[
                io.Autogrow.Input("dims", template=template,
                                  tooltip="Connect an integer per dimension, in order."),
                io.String.Input("prefix", default="B", advanced=True,
                                tooltip="Optional leading dimension, usually the batch name. "
                                        "Leave empty to use only the connected numbers."),
            ],
            outputs=[ShapeType.Output(display_name="shape")],
        )

    @classmethod
    def execute(cls, dims=None, prefix: str = "B") -> io.NodeOutput:
        values = [v for v in (dims or {}).values() if v is not None]
        parts = ([prefix.strip()] if prefix and prefix.strip() else []) + [int(v) for v in values]
        if not parts:
            raise ParseError(
                "No dimensions connected.",
                hint="Connect at least one integer, or type a name in 'prefix'.",
            )
        shape = Shape([_dim(p) if isinstance(p, str) else p for p in parts])
        return io.NodeOutput(shape, ui=ui.PreviewText(str(shape)))


class NeuroShapeInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroShapeInfo",
            display_name="Shape Info",
            category=CAT,
            description="Reads a shape as text and numbers, for wiring into other nodes or "
                        "just for looking at.",
            search_aliases=["shape to string", "rank", "count values"],
            inputs=[ShapeType.Input("shape")],
            outputs=[
                io.String.Output(display_name="text"),
                io.Int.Output(display_name="rank"),
                io.Int.Output(display_name="values per example"),
            ],
        )

    @classmethod
    def execute(cls, shape: Shape) -> io.NodeOutput:
        try:
            per_example = shape.numel(skip_leading=1)
        except Exception:
            per_example = 0
        return io.NodeOutput(str(shape), shape.rank, per_example,
                             ui=ui.PreviewText(f"{shape}\nrank {shape.rank}, "
                                               f"{per_example or 'unknown'} values per example"))


class NeuroInput(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroInput",
            display_name="Input",
            category=category("model"),
            description="Where the data enters the network. Everything downstream of this is "
                        "the architecture.\n\nThe shape can be typed here or connected from a "
                        "Shape node — or straight from a Dataset, which already knows what "
                        "shape its examples are.",
            search_aliases=["input layer", "placeholder", "entry", "x", "start here"],
            inputs=[
                io.MultiType.Input(
                    io.String.Input("shape", default="B, 2", tooltip=_SHAPE_HELP),
                    types=[ShapeType],
                ),
                io.String.Input("name", default="x",
                                tooltip="What this input is called in the summary and the "
                                        "exported code."),
                io.Combo.Input("dtype", options=list(DTYPES), default="float32",
                               tooltip="float32 for ordinary data. int64 for token ids going "
                                       "into an Embedding.", advanced=True),
            ],
            outputs=[Tensor.Output(display_name="tensor",
                                   tooltip="The data at the start of the network.")],
        )

    @classmethod
    def execute(cls, shape, name: str = "x", dtype: str = "float32") -> io.NodeOutput:
        parsed = to_shape(shape)
        tensor = make_input(parsed, dtype=dtype, name=(name or "x").strip())
        return io.NodeOutput(tensor, ui=badge_output(tensor))


def _dim(text: str) -> Dim:
    text = str(text).strip()
    if not text:
        raise ParseError("Empty dimension.", hint="Write a number, or a name like B.")
    return Dim(int(text)) if text.lstrip("-").isdigit() else Dim(text)


SHAPE_NODES = [NeuroShape, NeuroShapeImage, NeuroShapeSequence, NeuroShapeVector,
               NeuroShapeFromDims, NeuroShapeInfo, NeuroInput]
