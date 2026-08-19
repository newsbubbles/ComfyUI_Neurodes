"""One node that can be any single-input layer.

The discrete layer nodes are better on a canvas — you can read the architecture without
clicking anything. This one is for the other half of the job: trying things. Swapping a
Conv 2D for a Residual Block, or ReLU for GELU, is a dropdown away instead of a rewire, so
a comparison costs seconds.

Only layers that take exactly one tensor appear here. The two-input ones — Add, Concat,
Matrix Multiply — are the shape of the network rather than a step in it, and they deserve
to be visible as their own nodes.
"""

from __future__ import annotations

from comfy_api.latest import io

from ..core.compile import parameter_count
from ..core.registry import LayerSpec, all_specs, apply_layer
from ._helpers import LABEL_INPUT, SHARE_INPUT, badge_output, widgets
from .types import Tensor, category

#: display name -> spec, for the layers this node can be.
_CHOICES: dict[str, LayerSpec] = {
    spec.display: spec for spec in all_specs() if spec.arity == 1
}


def _options() -> list[io.DynamicCombo.Option]:
    return [io.DynamicCombo.Option(display, widgets(spec.params))
            for display, spec in _CHOICES.items()]


class NeuroLayerAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroLayerAny",
            display_name="Layer",
            category=category("layers"),
            description="Any single-input layer, chosen from a dropdown. The settings change "
                        "to match what you pick.\n\nUse this while exploring, when you want "
                        "to try five different layers in the same slot. Use the individual "
                        "nodes when you want the finished graph to be readable.",
            search_aliases=["any layer", "layer", "generic", "swap", "try", "experiment"],
            inputs=[
                Tensor.Input("x", tooltip="The tensor going into this layer."),
                io.DynamicCombo.Input(
                    "kind", options=_options(),
                    tooltip="Which layer this is. The settings below follow your choice."),
                SHARE_INPUT,
                LABEL_INPUT,
            ],
            outputs=[Tensor.Output(display_name="tensor")],
        )

    @classmethod
    def execute(cls, x, kind: dict, share: str = "", label: str = "") -> io.NodeOutput:
        chosen = kind["kind"]
        spec = _CHOICES[chosen]
        cfg = {p.name: kind.get(p.name, p.default) for p in spec.params}
        out = apply_layer(spec.key, [x], cfg, share=share or "", label=label or chosen)
        return io.NodeOutput(out, ui=badge_output(out, parameter_count(out.producer)))


ANY_LAYER_NODES = [NeuroLayerAny]
