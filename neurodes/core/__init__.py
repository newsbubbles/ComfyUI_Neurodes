"""neurodes core — everything that does not know ComfyUI exists.

Importing this module registers the whole layer library and gives you the public surface
the ComfyUI node adapters (and the test harness) use.
"""

from .errors import BuildError, GraphError, NeurodesError, ParseError, ShapeError
from .registry import (LayerSpec, Param, VARIADIC, all_specs, apply_layer, categories,
                       get, in_category)
from .shape import Dim, Shape, broadcast, unify
from .trace import Op, SymTensor, graph_inputs, make_input, topo_ops

from . import layers  # noqa: F401  side effect: registers every layer

from .compile import CompiledModel, build_model
from .emit import emit_source
from .summary import summarize

__all__ = [
    "NeurodesError", "ShapeError", "ParseError", "GraphError", "BuildError",
    "Dim", "Shape", "unify", "broadcast",
    "Op", "SymTensor", "make_input", "topo_ops", "graph_inputs",
    "LayerSpec", "Param", "VARIADIC", "get", "all_specs", "categories", "in_category",
    "apply_layer",
    "build_model", "CompiledModel", "emit_source", "summarize",
]
