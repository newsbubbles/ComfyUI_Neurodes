"""Every ComfyUI node in the pack, gathered in one list."""

from __future__ import annotations

from comfy_api.latest import io

from .any_layer import ANY_LAYER_NODES
from .data_nodes import DATA_NODES
from .layer_nodes import LAYER_NODES
from .model_nodes import MODEL_NODES
from .shape_nodes import SHAPE_NODES
from .train_nodes import TRAIN_NODES
from .vision_nodes import VISION_NODES
from .viz_nodes import VIZ_NODES

ALL_NODES: list[type[io.ComfyNode]] = (
    SHAPE_NODES + LAYER_NODES + ANY_LAYER_NODES + MODEL_NODES
    + DATA_NODES + TRAIN_NODES + VIZ_NODES + VISION_NODES
)

__all__ = ["ALL_NODES"]
