"""Diffusion: the training problem, and the loop that samples from it.

A diffusion model is the thing everyone using ComfyUI runs a hundred times a day and has
never been able to open up. It is also, stripped of the engineering, remarkably small:

1. Take a clean picture and mix it with noise, by an amount ``t`` between nothing and
   everything.
2. Train an ordinary network to look at the mess and say **which part of it was noise**.
3. To make a new picture, start from pure noise and repeatedly ask the network what the
   noise was, take a little of it away, and go round again.

Step 2 is a plain supervised problem — input, target, mean squared error — which is why a
U-Net trained by the same Train node as everything else in this pack turns into a generative
model. Nothing about the training is special. All the magic is in step 3, and step 3 is a
for loop.

**Where the timestep goes.** The network cannot do its job without knowing how noisy the
input is: the same grey smudge means different things at t=0.1 and t=0.9. Real
implementations feed the timestep in as a second argument and add its embedding to every
block. Train here hands a model one tensor, so the timestep rides along as **extra input
channels** — a couple of planes holding a Fourier encoding of t, concatenated onto the
image. It is a real technique rather than a workaround, and it has the pedagogical
advantage of being visible: you can see the timestep arrive on the wire as two more
channels, instead of it appearing by magic inside every residual block.
"""

from __future__ import annotations

import math

import torch

from .compile import CompiledModel
from .data import DataBundle
from .errors import NeurodesError
from .runtime import allocating

PREDICTS = ("noise", "image")
SCHEDULES = ("cosine", "linear")
TIMESTEPS = ("extra channels", "second input")


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------

def alpha_bar(t: torch.Tensor, schedule: str = "cosine") -> torch.Tensor:
    """How much of the *original picture* survives at time ``t``.

    ``t`` runs 0 (untouched) to 1 (pure noise), and the returned value runs 1 down to 0.
    The mixture is ``x_t = sqrt(a) * clean + sqrt(1 - a) * noise``, which keeps the total
    variance roughly constant however far along you are — that is the whole reason for the
    square roots.

    The cosine schedule spends more of its length near the clean end, where the differences
    that matter are, rather than destroying the picture in the first few steps.
    """
    if schedule == "cosine":
        s = 0.008
        f = torch.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2
        f0 = math.cos(s / (1.0 + s) * math.pi / 2) ** 2
        return (f / f0).clamp(1e-4, 1.0)
    if schedule == "linear":
        return (1.0 - t).clamp(1e-4, 1.0)
    raise NeurodesError(f"Unknown noise schedule {schedule!r}",
                        hint="Choose one of: " + ", ".join(SCHEDULES))


def time_features(t: torch.Tensor, channels: int) -> torch.Tensor:
    """The timestep as a small vector, ``[B, channels]``.

    A Fourier encoding rather than the raw number: ``sin`` and ``cos`` at doubling
    frequencies, so nearby timesteps look similar and distant ones look different, which is
    a much easier signal to use than a single flat value of ``0.63``.
    """
    if channels <= 0:
        return t.new_zeros((t.shape[0], 0))
    out = []
    for k in range((channels + 1) // 2):
        w = (2 ** k) * math.pi
        out.append(torch.sin(w * t))
        out.append(torch.cos(w * t))
    return torch.stack(out[:channels], dim=1).reshape(t.shape[0], -1)


def time_planes(t: torch.Tensor, channels: int, height: int, width: int) -> torch.Tensor:
    """The same encoding, spread over the picture so it can be stapled on as channels."""
    features = time_features(t, channels)
    if channels <= 0:
        return t.new_zeros((t.shape[0], 0, height, width))
    return features.view(-1, channels, 1, 1).expand(-1, -1, height, width).contiguous()


def add_noise(clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
              schedule: str = "cosine") -> torch.Tensor:
    """Mix a clean batch with noise by the amount the schedule says for each ``t``."""
    a = alpha_bar(t, schedule).view(-1, 1, 1, 1)
    return a.sqrt() * clean + (1.0 - a).sqrt() * noise


# ---------------------------------------------------------------------------
# Turning a folder of pictures into a diffusion training set
# ---------------------------------------------------------------------------

def as_diffusion_task(bundle: DataBundle, copies: int = 4, time_channels: int = 2,
                      schedule: str = "cosine", predict: str = "noise",
                      timestep: str = "extra channels", seed: int = 0) -> DataBundle:
    """Any image dataset becomes the supervised problem behind a diffusion model.

    Each picture is used ``copies`` times at different noise levels, because the network has
    to be good at every point on the journey, not just the middle. Labels are ignored
    entirely — like the image tasks, the supervision is manufactured, so a folder of
    pictures is already a complete training set.

    ``timestep`` decides how the network is told how noisy its input is. As
    ``extra channels`` the encoding is stapled onto the picture, which needs nothing of the
    model but one wider Input. As a ``second input`` it arrives as its own small vector, the
    way a real implementation does it — and then the graph has to say what to do with it,
    which is the more honest and more interesting version.
    """
    if bundle.x_train.dim() != 4:
        raise NeurodesError(
            f"Diffusion needs pictures, but this dataset is {bundle.input_shape}.",
            hint="Use an Image Dataset or an Image Folder Dataset. Diffusion works on "
                 "[batch, channels, height, width].")
    if predict not in PREDICTS:
        raise NeurodesError(f"Unknown prediction target {predict!r}",
                            hint="Choose one of: " + ", ".join(PREDICTS))
    if timestep not in TIMESTEPS:
        raise NeurodesError(f"Unknown timestep placement {timestep!r}",
                            hint="Choose one of: " + ", ".join(TIMESTEPS))
    alpha_bar(torch.zeros(1), schedule)                     # validates the schedule name

    channels = int(time_channels)
    copies = max(1, int(copies))
    separate = timestep == "second input"

    with allocating():
        g = torch.Generator().manual_seed(int(seed))

        def build(clean: torch.Tensor, repeats: int):
            if clean.shape[0] == 0:
                width = clean.shape[1] + (0 if separate else channels)
                return (clean.new_zeros((0, width, *clean.shape[2:])),
                        clean.new_zeros((0, channels)), clean.clone())
            clean = clean.repeat(repeats, 1, 1, 1)
            n, _, h, w = clean.shape
            t = torch.rand((n, 1), generator=g).to(clean)
            noise = torch.randn(clean.shape, generator=g).to(clean)
            noisy = add_noise(clean, t, noise, schedule)
            features = time_features(t, channels)
            x = noisy if separate else torch.cat([noisy, time_planes(t, channels, h, w)], dim=1)
            return x, features, (noise if predict == "noise" else clean)

        x_train, t_train, y_train = build(bundle.x_train, copies)
        # One noise level per validation image: the score should not move because the
        # validation set was re-rolled, so it is drawn once and left alone.
        x_val, t_val, y_val = build(bundle.x_val, 1)

    picture_channels = int(bundle.x_train.shape[1])
    where = (f"a second input of {channels} number(s)" if separate
             else f"{channels} extra channel(s)")
    return DataBundle(
        x_train, y_train, x_val, y_val, task="reconstruction", classes=(),
        name=f"{bundle.name} (diffusion)",
        notes=f"Each picture appears at {copies} noise level(s). The model is given "
              f"{picture_channels} noisy channel(s) and the timestep as {where}, and has to "
              f"produce the {predict}.",
        side_train=(t_train,) if separate else (),
        side_val=(t_val,) if separate else (),
        extra={"diffusion": {"schedule": schedule, "predict": predict,
                             "time_channels": channels, "channels": picture_channels,
                             "timestep": timestep,
                             "size": (int(bundle.x_train.shape[2]),
                                      int(bundle.x_train.shape[3]))}},
    )


def config_of(bundle: DataBundle) -> dict:
    """The diffusion settings a dataset was built with, or a clear complaint."""
    cfg = (bundle.extra or {}).get("diffusion")
    if not cfg:
        raise NeurodesError(
            f"{bundle.name} is not a diffusion dataset.",
            hint="Put a Dataset As Diffusion node between the dataset and this one. The "
                 "sampler reads the noise schedule from it, so the settings cannot drift "
                 "apart from the ones the model was trained on.")
    return cfg


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model: CompiledModel, cfg: dict, count: int = 8, steps: int = 40,
           size: tuple[int, int] | None = None, seed: int = 0,
           keep_trajectory: bool = False, guidance: float = 0.0):
    """Run the loop backwards: pure noise in, pictures out.

    This is DDIM — deterministic, and good at few steps, which matters when the whole point
    is to watch it happen. Each iteration asks the model what it thinks the noise is, uses
    that to guess the clean picture, and then re-noises the guess to the *next* timestep
    down. The guesses at the start are terrible; the last twenty per cent is where the
    picture appears.

    Returns the finished batch, and optionally every intermediate step as one long batch —
    which is a Video Combine away from a clip of the thing forming.
    """
    schedule = cfg.get("schedule", "cosine")
    predict = cfg.get("predict", "noise")
    time_channels = int(cfg.get("time_channels", 2))
    channels = int(cfg.get("channels", 3))
    h, w = size or tuple(cfg.get("size", (32, 32)))
    steps = max(1, int(steps))
    count = max(1, int(count))

    device = model.device
    model.eval()
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn((count, channels, h, w), generator=g).to(device)

    grid = torch.linspace(1.0, 0.0, steps + 1, device=device)
    trajectory = [] if keep_trajectory else None

    separate = cfg.get("timestep", "extra channels") == "second input"
    for i in range(steps):
        t_now = grid[i].expand(count, 1)
        if separate:
            out = model(x, time_features(t_now, time_channels))
        else:
            out = model(torch.cat([x, time_planes(t_now, time_channels, h, w)], dim=1))
        x, clean, _ = step(x, out, t_now, grid[i + 1].expand(count, 1),
                           schedule=schedule, predict=predict, guidance=guidance)
        if trajectory is not None:
            trajectory.append(clean.detach().cpu().clone())

    final = x.clamp(0, 1).detach().cpu()
    return (final, torch.cat(trajectory) if trajectory else final)


def step(x: torch.Tensor, out: torch.Tensor, t_now: torch.Tensor, t_next: torch.Tensor,
         schedule: str = "cosine", predict: str = "noise", guidance: float = 0.0):
    """One turn of the loop: where we are, what the model said, where we go next.

    Returns ``(x_next, clean, noise)``. The invariant that makes the loop work is that
    ``clean`` and ``noise`` must be a valid decomposition of the picture we are *holding* —
    ``x == sqrt(a_now) * clean + sqrt(1 - a_now) * noise`` — because the next line assumes
    exactly that and re-mixes them at ``t_next``.
    """
    a_now = alpha_bar(t_now, schedule).view(-1, 1, 1, 1)
    a_next = alpha_bar(t_next, schedule).view(-1, 1, 1, 1)

    clean = (x - (1.0 - a_now).sqrt() * out) / a_now.sqrt() if predict == "noise" else out

    if guidance:
        # Push away from the average of the batch. Not classifier-free guidance -- there is
        # nothing to condition on here -- but the same trick of exaggerating what makes this
        # sample different, and it sharpens an undertrained model.
        clean = clean + guidance * (clean - clean.mean(dim=0, keepdim=True))

    # Pictures live in [0, 1], so a guess outside it is known to be wrong -- and near t=1
    # the division above multiplies the model's error by a hundred, so the guesses there are
    # very wrong indeed. Clip, then rebuild the noise term *from the clipped guess*, which is
    # what keeps the invariant above true. Skipping that second half leaves the two terms
    # describing different pictures; the disagreement compounds, and more sampling steps make
    # the result worse rather than better.
    clean = clean.clamp(0, 1)
    noise = (x - a_now.sqrt() * clean) / (1.0 - a_now).clamp_min(1e-8).sqrt()
    return a_next.sqrt() * clean + (1.0 - a_next).sqrt() * noise, clean, noise
