"""One ComfyUI node per registered layer, generated from the registry.

Sixty-odd nodes and no boilerplate. More importantly: the widgets, the shape inference,
the module that gets built and the code that gets exported all come from the same
declaration, so they cannot disagree with each other.
"""

from __future__ import annotations

from comfy_api.latest import io

from ..core.compile import parameter_count
from ..core.registry import LayerSpec, VARIADIC, all_specs, apply_layer
from ._helpers import (LABEL_INPUT, SHARE_INPUT, badge_output, camel, collect_tensors,
                       tensor_inputs, widgets)
from .types import Tensor, category


def _describe(spec: LayerSpec) -> str:
    """Node tooltip: what it does, then what it eats."""
    lines = [spec.doc.strip()]
    if spec.arity == VARIADIC:
        lines.append("\nTakes any number of tensors.")
    elif spec.arity > 1:
        lines.append("\nTakes " + ", ".join(spec.input_label(i) for i in range(spec.arity)) + ".")
    return "\n".join(x for x in lines if x)


def make_layer_node(spec: LayerSpec) -> type[io.ComfyNode]:
    node_id = f"NeuroLayer{camel(spec.key)}"

    class LayerNode(io.ComfyNode):
        SPEC = spec

        @classmethod
        def define_schema(cls) -> io.Schema:
            s: LayerSpec = cls.SPEC
            return io.Schema(
                node_id=node_id,
                display_name=s.display,
                category=category(s.category),
                description=_describe(s),
                search_aliases=list(s.aliases),
                inputs=tensor_inputs(s) + widgets(s.params) + [SHARE_INPUT, LABEL_INPUT],
                outputs=[Tensor.Output(display_name="tensor",
                                       tooltip="The tensor after this layer, with its new shape.")],
            )

        @classmethod
        def execute(cls, **kwargs) -> io.NodeOutput:
            s: LayerSpec = cls.SPEC
            tensors = collect_tensors(s, kwargs)
            cfg = {p.name: kwargs.get(p.name, p.default) for p in s.params}
            out = apply_layer(s.key, tensors, cfg,
                              share=kwargs.get("share", "") or "",
                              label=kwargs.get("label", "") or "")
            return io.NodeOutput(out, ui=badge_output(out, parameter_count(out.producer)))

    LayerNode.__name__ = node_id
    LayerNode.__qualname__ = node_id
    LayerNode.__doc__ = spec.doc
    return LayerNode


#: Built once at import; the entrypoint hands this straight to ComfyUI.
LAYER_NODES: list[type[io.ComfyNode]] = [make_layer_node(spec) for spec in all_specs()]
