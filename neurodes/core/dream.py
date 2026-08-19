"""Deep dream and feature visualisation.

Training asks: given this input, what should the weights be? This asks the question the
other way round: given these weights, what input would excite them most? Answering it is
the same machinery — a forward pass, a loss, a backward pass — with the gradient applied to
the picture instead of to the network.

That inversion is why this is worth having in a teaching pack and not only a pretty one.
"What did it learn?" stops being an article of faith and becomes an image you generated
from the weights themselves.

Two things make the difference between noise and the recognisable look:

**Octaves.** Optimise a small canvas, scale it up, optimise again. Low frequencies get
settled while the picture is small and cheap, and each upscale gives the next round
somewhere to put detail. Skipping this gives high-frequency confetti.

**Jitter.** Roll the image by a few pixels before every step. The gradient can then never
sharpen a feature onto one fixed pixel grid, which is the other half of what causes
confetti.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .errors import BuildError, NeurodesError
from .runtime import trainable

OBJECTIVES = ("mean", "l2 norm", "max")


def _score(activation: torch.Tensor, channel: int, objective: str) -> torch.Tensor:
    """How excited the chosen part of the network is. Gradient ascent maximises this."""
    if channel >= 0:
        if activation.dim() < 2:
            raise NeurodesError(
                "This layer has no channel dimension to pick from.",
                hint="Set channel to -1 to excite the whole layer.",
            )
        if channel >= activation.shape[1]:
            raise NeurodesError(
                f"Channel {channel} does not exist: this layer has {activation.shape[1]}.",
                hint=f"Use 0 to {activation.shape[1] - 1}, or -1 for the whole layer.",
            )
        activation = activation[:, channel]
    if objective == "l2 norm":
        return activation.pow(2).mean()
    if objective == "max":
        return activation.amax()
    return activation.mean()


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Penalises neighbouring pixels differing, which smooths out the speckle."""
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


_BLUR = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])


def _smooth(t: torch.Tensor) -> torch.Tensor:
    """Separable 5-tap blur, applied to the *gradient* rather than the picture.

    Left alone, gradient ascent finds that the cheapest way to excite an edge detector is a
    one-pixel-wide grating, and the result is uniform diagonal hatching that tells you
    nothing. Blurring the gradient before applying it removes that shortcut, so the
    optimiser has to build features at a scale you can actually see. This is the single
    biggest difference between deep dream output that looks like something and deep dream
    output that looks like corduroy.
    """
    channels = t.shape[1]
    kernel = (_BLUR / _BLUR.sum()).to(t)
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    t = F.conv2d(F.pad(t, (2, 2, 0, 0), mode="reflect"), horizontal, groups=channels)
    return F.conv2d(F.pad(t, (0, 0, 2, 2), mode="reflect"), vertical, groups=channels)


def dream(model, layer: str, canvas: torch.Tensor, *, steps: int = 20,
          learning_rate: float = 0.006, channel: int = -1, objective: str = "mean",
          octaves: int = 3, octave_scale: float = 1.4, jitter: int = 4,
          tv_weight: float = 0.0, feature_scale: float = 0.9, clamp: bool = True,
          on_step=None, should_stop=None) -> torch.Tensor:
    """Push ``canvas`` uphill until ``layer`` is as excited as it can make it.

    ``canvas`` is ``[B, C, H, W]`` in the model's own input space. Returns the same shape.
    """
    model.eval()

    # Deliberately NOT touching parameter.requires_grad here. ComfyUI caches node outputs,
    # so this is the very same model object a Train node upstream will be handed on the
    # next run — switching its gradients off would leave the network permanently
    # untrainable, and the failure would surface over in Train with no hint of what did it.
    # torch.autograd.grad(loss, x) returns only the gradient with respect to x and never
    # accumulates into the parameters, so there was nothing to gain by it anyway.

    # A layer this model does not have should say so before any work happens.
    model._step_named(layer)

    base_h, base_w = canvas.shape[-2], canvas.shape[-1]
    octaves = max(1, int(octaves))
    total_steps = steps * octaves
    done = 0

    with trainable():
        # Training will have left the model on the GPU; the canvas has to follow it there.
        x = canvas.detach().clone().to(model.device)
        for octave in range(octaves):
            scale = octave_scale ** (octaves - 1 - octave)
            size = (max(4, int(base_h / scale)), max(4, int(base_w / scale)))
            if x.shape[-2:] != size:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            x = x.detach().requires_grad_(True)

            for step in range(steps):
                if should_stop is not None and should_stop():
                    return x.detach()

                shifted = x
                if jitter > 0:
                    dy = int(torch.randint(-jitter, jitter + 1, (1,)).item())
                    dx = int(torch.randint(-jitter, jitter + 1, (1,)).item())
                    shifted = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))

                try:
                    activation = model.forward_to(layer, shifted)
                except RuntimeError as exc:
                    size_related = any(word in str(exc).lower() for word in
                                       ("shape", "size", "mat1", "mat2", "dimension"))
                    raise BuildError(
                        f"Could not run the network up to '{layer}' at "
                        f"{size[0]}x{size[1]}: {exc}",
                        hint=("Layers after a Flatten are locked to the size the model was "
                              "built for. Pick a convolutional layer before the Flatten, or "
                              "set the canvas to the model's own input size."
                              if size_related else
                              "This is not a size problem, so it is a bug in neurodes rather "
                              "than in your workflow."),
                    ) from None

                loss = _score(activation, channel, objective)
                if tv_weight:
                    loss = loss - tv_weight * _total_variation(shifted)
                grad, = torch.autograd.grad(loss, x)

                if feature_scale > 0:
                    grad = grad * (1 - feature_scale) + _smooth(grad) * feature_scale

                # Normalising by mean absolute value, not standard deviation: it is what the
                # original DeepDream used, and it is far less twitchy when most of the
                # gradient is zero, which after a ReLU it usually is.
                grad = grad / grad.abs().mean().clamp(min=1e-8)
                with torch.no_grad():
                    x += learning_rate * grad
                    if clamp:
                        x.clamp_(0.0, 1.0)

                done += 1
                if on_step is not None:
                    on_step(done, total_steps, {"octave": octave + 1, "score": loss.item()})

        if x.shape[-2:] != (base_h, base_w):
            x = F.interpolate(x, size=(base_h, base_w), mode="bilinear", align_corners=False)
    return x.detach()


def noise_canvas(shape, batch: int = 1, size: int = 0, seed: int = 0,
                 blur: bool = True) -> torch.Tensor:
    """A starting picture when the user supplies none.

    Slightly blurred low-contrast noise, rather than white noise, because gradient ascent
    from white noise spends its first hundred steps undoing the noise.
    """
    channels = shape[1].size if shape.rank == 4 and shape[1].is_concrete else 3
    if size > 0:
        height = width = size
    elif shape.rank == 4 and shape[2].is_concrete and shape[3].is_concrete:
        height, width = shape[2].size, shape[3].size
    else:
        height = width = 256
    generator = torch.Generator().manual_seed(int(seed))
    x = torch.rand(batch, channels, height, width, generator=generator) * 0.2 + 0.4
    if blur:
        kernel = torch.ones(channels, 1, 3, 3) / 9.0
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel, groups=channels)
    return x


def dreamable_layers(model) -> list[str]:
    """The layers worth pointing this at: the ones that still have spatial extent.

    A Linear layer's activation is a vector, so maximising it produces a picture with no
    structure to look at. Convolutional layers are where the recognisable patterns live.
    """
    from .emit import shapes_by_op

    shapes = shapes_by_op(model.outputs)
    return [model.step_names[step.op.uid] for step in model.plan
            if shapes.get(step.op.uid) is not None and shapes[step.op.uid].rank == 4]
