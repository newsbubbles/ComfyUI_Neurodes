"""The socket types.

This is the smallest file in the pack and the most important idea in it. Declaring these
is what makes ComfyUI refuse to connect a dataset to a layer, or a shape to a loss — the
rules of neural network construction, enforced by the editor, for free.
"""

from __future__ import annotations

from comfy_api.latest import io

#: A tensor as it is known at design time: a shape, a dtype, and where it came from.
Tensor = io.Custom("NEURO_TENSOR")

#: A list of dimensions, possibly with named ones.
ShapeType = io.Custom("NEURO_SHAPE")

#: A built, runnable model. Carries weights.
Model = io.Custom("NEURO_MODEL")

#: Training and validation data, already split.
Dataset = io.Custom("NEURO_DATASET")

#: What happened during training: losses, accuracies, timings, notes.
History = io.Custom("NEURO_HISTORY")

#: Optimizer, loss and hyperparameters, bundled so one recipe can drive several runs.
Trainer = io.Custom("NEURO_TRAINER")

#: Every intermediate tensor from one forward pass, keyed by layer name. Captured once so
#: that viewing five layers costs one pass, not five.
Activations = io.Custom("NEURO_ACTIVATIONS")

CATEGORY = "neurodes"


def category(*parts: str) -> str:
    return "/".join((CATEGORY,) + parts)
