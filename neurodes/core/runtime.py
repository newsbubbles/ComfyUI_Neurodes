"""Escaping the host's inference mode.

ComfyUI executes every prompt inside ``torch.inference_mode()``. That is exactly right for
its usual job — running a diffusion model as fast as possible — and exactly wrong for ours.

Inference mode is stronger than ``no_grad``. It is not merely "do not record gradients": a
tensor *created* while it is active is permanently marked as an inference tensor and can
never take part in autograd, even after the mode is exited. A model built under it would
have weights that can never be trained, and data allocated under it cannot be saved for
backward.

So anything that allocates a tensor we will later differentiate through has to step outside
first. Outside ComfyUI these context managers do nothing of consequence, which is why they
live in core rather than in the node layer.
"""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def trainable():
    """Allocate tensors that autograd can actually use, and record gradients."""
    with torch.inference_mode(False), torch.enable_grad():
        yield


@contextlib.contextmanager
def allocating():
    """Allocate real (non-inference) tensors, without turning gradient recording on."""
    with torch.inference_mode(False), torch.no_grad():
        yield


def adopt(tensor: torch.Tensor) -> torch.Tensor:
    """Take a tensor that may have come from the host and make it safe to train with.

    A tensor created under inference mode cannot be saved for backward, so passing one in
    as training data fails at the first ``loss.backward()``. Cloning it outside the mode
    produces an ordinary tensor. The clone is skipped when it is not needed.
    """
    if not torch.is_tensor(tensor):
        return tensor
    if not getattr(tensor, "is_inference", lambda: False)():
        return tensor
    with allocating():
        return tensor.clone()
