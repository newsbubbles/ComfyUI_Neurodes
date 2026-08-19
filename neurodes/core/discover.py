"""Learning convolution kernels with nobody to say what they should be.

Every other training node in the pack answers a question of the form "here is the right
answer, get closer to it". This one has no right answer. It has an image, a bank of empty
kernels, and a statement about what a *good* kernel does — and it goes looking.

The tempting first objective is "make the output histogram as different from the input's as
possible". It does not work, and the way it fails is worth seeing rather than being told
about, so ``histogram change`` is one of the objectives you can pick. What it learns from a
photograph is sixteen tiles of static that are 94% identical to one another. Two things are
wrong with it and both are instructive:

**It is not blind to scale.** The score changes when you multiply the response by a
constant, so the optimiser optimises the constant and leaves the filter alone. Which way it
runs depends on the picture, and neither way is good. On mean-centred patches the cheapest
histogram that is maximally unlike a photograph's is a single spike at zero, so the bank
switches itself *off* — measured on a beach photograph, a response 26 times *smaller* than
the input, where the working objective's is 1.3 times its size. A filter bank whose best
available move is to output nothing at all. Add the diversity term and it bolts the other
way instead, to a response 75 times too large. Either way the thing being trained is the
volume knob, and the sparsity it ends on is 0.619 where untrained kernels score 0.621.

The fix is to score a *scale-invariant* statistic, one whose value is unchanged if you
multiply the response by anything. Then the volume earns nothing and the only way left to
move the score is to change the *shape* of the distribution, which was the interesting part
in the first place.

**It cannot tell sixteen kernels from one kernel sixteen times.** Every kernel is scored on
its own, so they all walk downhill to the same minimum. This one is not a flaw in the
objective so much as something the objective is silent about, and choosing a better
statistic does not fix it — the good objective collapses too, to a mean overlap of 0.867
between neighbouring kernels. Worse, the collapsed bank *scores better*: 0.077 of sparsity
gained against 0.069 for the diverse one. Of course it does. It found the single best
filter and made sixteen copies, and sixteen copies of the best answer is sixteen good
answers as far as the objective can see. The loss curve is never going to warn you about
this, and no amount of staring at the loss curve could. Kernels have to be told to
disagree, which is what ``diversity`` does: it penalises correlation between what any two
of them say about the same patch.

With both fixed, what comes out is not arbitrary. Oriented edge detectors at a range of
angles and scales, localised, band-pass — the same small alphabet that falls out of sparse
coding (Olshausen & Field 1996) and of ICA on natural scenes (Bell & Sejnowski 1997), and
much the same as the first layer of any convolutional net trained on photographs, and much
the same as what has been measured in mammalian primary visual cortex. Nobody said the word
"edge" anywhere in the graph. Edges are simply what is left when you ask for a filter whose
response to a photograph is *rarely* large.

Change the picture and the alphabet changes with it, which is the part worth playing with.
"""

from __future__ import annotations

import math

import torch

from .data import DataBundle
from .errors import NeurodesError

OBJECTIVES = ("sparse response", "peaky response", "histogram change")

GAUSSIAN_L1L2 = math.sqrt(2.0 / math.pi)
"""E|z| / sqrt(E z^2) for a standard normal: 0.7979.

The reference every sparsity number here is read against. A filter whose response to an
image scores near this is telling you it found nothing — a weighted sum of a hundred pixels
is Gaussian by the central limit theorem unless the weights are doing something specific.
"""

GAUSSIAN_KURTOSIS = 3.0


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

def _planes(images: torch.Tensor, greyscale: bool) -> torch.Tensor:
    """ComfyUI's [batch, height, width, channels] to torch's [batch, channels, h, w]."""
    x = images.detach().float()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise NeurodesError(f"Expected an image batch, got a tensor shaped {tuple(x.shape)}.")
    x = x.permute(0, 3, 1, 2) if x.shape[-1] <= 4 else x
    if greyscale and x.shape[1] >= 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
        x = (x[:, :3] * weights).sum(dim=1, keepdim=True)
    elif greyscale and x.shape[1] > 1:
        x = x[:, :1]
    return x.clamp(0, 1)


def _cut(region: torch.Tensor, patch: int, count: int, generator) -> torch.Tensor:
    """``count`` patches from random positions in ``region``, shaped [B, C, H, W]."""
    b, _, h, w = region.shape
    if h < patch or w < patch:
        raise NeurodesError(
            f"A {patch}x{patch} patch does not fit in a {h}x{w} region of the image.",
            hint="Use a smaller patch size, or a larger image.",
        )
    which = torch.randint(0, b, (count,), generator=generator)
    ys = torch.randint(0, h - patch + 1, (count,), generator=generator)
    xs = torch.randint(0, w - patch + 1, (count,), generator=generator)
    return torch.stack([region[i, :, y:y + patch, x:x + patch]
                        for i, y, x in zip(which.tolist(), ys.tolist(), xs.tolist())])


def image_patches(images: torch.Tensor, patch: int = 12, count: int = 6000,
                  greyscale: bool = True, remove_mean: bool = True,
                  val_fraction: float = 0.15, seed: int = 0,
                  name: str = "image patches") -> DataBundle:
    """Cut random square patches out of an image, with no labels and nothing to predict.

    ``remove_mean`` subtracts each patch's own average brightness. It is not cosmetic. A
    patch's mean is by far the largest thing in it and a filter that simply measures
    brightness scores well on every objective here without having learned anything about
    structure; with the mean left in, kurtosis lands at 8.4 instead of 19.5 and the kernels
    come out visibly worse. The alternative cure is to force each kernel to sum to zero,
    which does the same job — you want one of the two, and doing both buys nothing measurable.

    Validation patches come from a held-out strip down the right-hand edge rather than from
    a random split of the same positions. Patches overlap heavily, so a random split would
    put a patch in validation that shares most of its pixels with one in training, and the
    validation curve would be measuring memory rather than generalisation.
    """
    x = _planes(images, greyscale)
    patch, count = max(2, int(patch)), max(16, int(count))
    generator = torch.Generator().manual_seed(int(seed))

    width = x.shape[3]
    keep = min(max(0.0, float(val_fraction)), 0.5)
    cut = int(width * (1.0 - keep))
    # Never let the split leave either side too thin to cut a patch from; below that, hand
    # the whole image to training and say so rather than failing.
    note = ""
    if cut < patch or width - cut < patch:
        cut, note = width, "the image is too narrow to hold out a validation strip"
    # A floor rather than purely a fraction of the training count. The held-out strip is a
    # region of the image, so cutting more patches out of it costs almost nothing, and the
    # validation set here is asked to estimate a fourth moment and a correlation between
    # every pair of kernels — both of which are badly served by a few hundred rows.
    n_val = max(2048, int(count * keep)) if cut < width else 256

    train = _cut(x[..., :cut], patch, count, generator)
    val = _cut(x[..., cut:] if cut < width else x, patch, n_val, generator)
    if remove_mean:
        train = train - train.mean(dim=(1, 2, 3), keepdim=True)
        val = val - val.mean(dim=(1, 2, 3), keepdim=True)

    # There is no target. A column of zeros keeps DataBundle's shape machinery working, and
    # every objective here ignores it.
    y_train = torch.zeros(train.shape[0], 1)
    y_val = torch.zeros(val.shape[0], 1)
    notes = f"{patch}x{patch} patches, " + ("mean removed" if remove_mean else "mean kept")
    if note:
        notes += f"; {note}"
    return DataBundle(train, y_train, val, y_val, task="discovery", name=name, notes=notes,
                      extra={"patch": patch, "greyscale": bool(greyscale),
                             "remove_mean": bool(remove_mean)})


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def responses(out: torch.Tensor) -> torch.Tensor:
    """Whatever the model produced, as [samples, kernels].

    A kernel the size of the whole patch gives one number per patch; a smaller one gives a
    little response map. Both are the same thing for scoring purposes — every position is
    another sample of what that kernel says about this image — so the spatial axes are
    folded into the sample axis.
    """
    if out.dim() == 2:
        return out
    if out.dim() < 2:
        raise NeurodesError(
            f"The model returns a rank-{out.dim()} tensor, which has no channels to score.",
            hint="This objective needs a layer that produces several feature maps, such as "
                 "a Conv 2D.",
        )
    return out.movedim(1, -1).reshape(-1, out.shape[1])


def soft_histogram(v: torch.Tensor, bins: int = 32, lo: float = 0.0,
                   hi: float = 1.0) -> torch.Tensor:
    """A histogram you can differentiate through.

    Counting into bins is a step function and has no gradient anywhere. Each value instead
    contributes a small bump to every bin, falling off with distance, which is the same
    picture in the limit and has a gradient everywhere.
    """
    centres = torch.linspace(lo, hi, bins, device=v.device, dtype=v.dtype)
    width = (hi - lo) / max(1, bins - 1)
    weight = torch.exp(-(((v.reshape(-1, 1) - centres.reshape(1, -1)) / width) ** 2))
    total = weight.sum(dim=0)
    return total / total.sum().clamp_min(1e-8)


def _centred(r: torch.Tensor) -> torch.Tensor:
    return r - r.mean(dim=0, keepdim=True)


def sparsity(r: torch.Tensor) -> torch.Tensor:
    """E|r| / sqrt(E r^2), per kernel. Lower is sparser. Gaussian is 0.798.

    Scale-invariant by construction: multiply the response by anything and both halves of
    the ratio scale with it. That is the whole point — it cannot be improved by turning up
    the gain, so the only way to move it is to change the shape of the distribution.
    """
    c = _centred(r)
    return c.abs().mean(dim=0) / c.pow(2).mean(dim=0).sqrt().clamp_min(1e-8)


def kurtosis(r: torch.Tensor) -> torch.Tensor:
    """E r^4 / (E r^2)^2, per kernel. Gaussian is 3. Also scale-invariant.

    Sharper than :func:`sparsity` at telling a spike from a bell, and correspondingly more
    easily dominated by a handful of outlying patches — which is why it is not the default.

    Divided through by the standard deviation *before* the fourth power rather than after.
    Written the direct way, a response around 1e-3 gives a fourth moment around 1e-12,
    which is exactly where the guard against dividing by zero sits — so the guard fires on
    ordinary data and the answer comes back quietly wrong. That is not hypothetical: the
    naive objective's favourite trick is driving the response very small.
    """
    c = _centred(r)
    c = c / c.pow(2).mean(dim=0).sqrt().clamp_min(1e-12)
    return c.pow(4).mean(dim=0)


def correlation(r: torch.Tensor) -> torch.Tensor:
    """The kernels' responses correlated against each other, [kernels, kernels]."""
    z = _centred(r)
    z = z / z.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (z.t() @ z) / max(1, z.shape[0])


def redundancy(r: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal correlation: how much the kernels repeat each other."""
    k = r.shape[1]
    if k < 2:
        return r.new_zeros(())
    corr = correlation(r)
    off = corr - torch.diag(torch.diag(corr))
    return off.pow(2).sum() / (k * (k - 1))


def objective(name: str, diversity: float = 0.0, reference: torch.Tensor | None = None):
    """Build the loss. Returns ``f(output, target)`` — the target is ignored throughout.

    ``diversity`` is added to every objective including the naive one, so the two failures
    can be separated: turning it on rescues the collapse without rescuing the gain problem,
    which is how you can tell they are two different faults and not one.
    """
    if name not in OBJECTIVES:
        raise NeurodesError(f"Unknown objective {name!r}",
                            hint="Choose one of: " + ", ".join(OBJECTIVES))
    weight = max(0.0, float(diversity))

    def loss(out, target=None):
        r = responses(out)
        if name == "sparse response":
            value = sparsity(r).mean()
        elif name == "peaky response":
            # log, not the raw ratio: kurtosis runs over orders of magnitude and the raw
            # value would let one lucky kernel drown out the other fifteen.
            value = -torch.log(kurtosis(r).clamp_min(1e-6)).mean()
        else:
            # The naive idea, implemented faithfully so that it fails for its own reasons
            # and not because it was hobbled. The response is read as if it were an image,
            # because "the histogram changed" is a statement about an image.
            here = soft_histogram(r.clamp(0.0, 1.0))
            value = -(here - reference.to(here)).abs().sum()
        return value + weight * redundancy(r) if weight else value

    loss.objective_name = name
    loss.diversity = weight
    return loss


def reference_histogram(data: DataBundle) -> torch.Tensor:
    """The input's own histogram, for ``histogram change`` to be different from."""
    return soft_histogram(data.x_train[:4096].reshape(-1).detach())


# ---------------------------------------------------------------------------
# What came out
# ---------------------------------------------------------------------------

def shape_at(model, layer: str = ""):
    """The shape a named layer produces, or the model's output shape when unnamed."""
    if not layer:
        return model.output_shapes[0]
    from .emit import shapes_by_op

    step = model._step_named(layer)
    return shapes_by_op(model.outputs).get(step.op.uid) or model.output_shapes[0]


def activation(model, layer: str, *inputs) -> torch.Tensor:
    return model.forward_to(layer, *inputs) if layer else model(*inputs)


def feeds(model, layer: str) -> str:
    """The step immediately before ``layer`` in execution order, or '' for the input.

    Needed so that "how loud is this layer" is measured against what actually arrives at
    it. Measured against the *model's* input instead, the ratio compounds with every layer
    — the third layer of a stack reads as 134x and looks like the runaway-gain failure when
    it is only the ordinary accumulation of three layers each a bit louder than its input.
    """
    if not layer:
        return ""
    names = [model.step_names[s.op.uid] for s in model.plan]
    index = names.index(layer) if layer in names else 0
    return names[index - 1] if index > 0 else ""


def weight_at(model, layer: str = "") -> tuple[str, torch.Tensor] | tuple[None, None]:
    """The kernels of a named layer, or of the first layer that has any."""
    if not layer:
        return first_weight(model)
    module = model.module_for(model._step_named(layer).op)
    weight = getattr(module, "weight", None) if module is not None else None
    return (layer, weight.detach()) if weight is not None else (None, None)


def kernel_overlap(weight: torch.Tensor) -> torch.Tensor:
    """|cosine| between every pair of kernels, diagonal zeroed."""
    flat = weight.detach().reshape(weight.shape[0], -1).float()
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (flat @ flat.t()).abs().fill_diagonal_(0)


def first_weight(model) -> tuple[str, torch.Tensor] | tuple[None, None]:
    """The first layer that has a weight — the kernels, for a one-convolution model."""
    for step in model.plan:
        module = model.module_for(step.op)
        weight = getattr(module, "weight", None) if module is not None else None
        if weight is not None and weight.dim() == 4:
            return step.name, weight.detach()
    return None, None


@torch.no_grad()
def untrained_floor(model, data, seeds: int = 3, limit: int = 4096, layer: str = "") -> dict:
    """What this same architecture scores *before* it learns anything.

    Necessary, not decorative. These statistics measure non-Gaussianity, and photographs are
    already extremely non-Gaussian before any filter touches them — the same random kernels
    that score 0.799 on Gaussian noise, exactly as theory says they must, score 0.659 on a
    beach photograph and 0.522 on a screenshot of text. So an absolute number means nothing
    on its own: a bank that learned nothing at all on the screenshot would post a better
    sparsity than a well-trained bank on the beach, and a report quoting the textbook
    Gaussian reference would present the picture's own statistics as an achievement.

    The floor is measured rather than assumed: the weights are re-initialised a few times,
    scored, and the real weights put back.
    """
    keep = {k: v.detach().clone() for k, v in model.state_dict().items()}
    x = (data.x_val if data.n_val >= 64 else data.x_train)[:limit].to(model.device)
    sp, ku = [], []
    try:
        for s in range(max(1, seeds)):
            model.reinitialise(seed=9000 + s, only=layer)
            r = responses(activation(model, layer, x)).float()
            sp.append(sparsity(r).mean().item())
            ku.append(kurtosis(r).mean().item())
    finally:
        model.load_state_dict(keep)
    spread = torch.tensor(sp)
    return {"sparsity": spread.mean().item(), "kurtosis": sum(ku) / len(ku),
            "sparsity_sd": spread.std().item() if len(sp) > 1 else 0.0}


@torch.no_grad()
def response_pair(model, data: DataBundle, kernel: int = -1, limit: int = 4096,
                  layer: str = "") -> tuple[torch.Tensor, torch.Tensor]:
    """The trained bank's responses and the untrained ones, for plotting side by side.

    ``kernel`` picks one filter; -1 pools them all.
    """
    model.eval()
    x = (data.x_val if data.n_val >= 64 else data.x_train)[:limit].to(model.device)

    def pick(out):
        r = responses(out).float()
        if kernel < 0:
            # Each kernel standardised *before* pooling. Without this the picture is a
            # mixture of differently-scaled bells, and a mixture of bells is heavy-tailed
            # whatever the bells are — untrained and trained banks then draw the same curve
            # while the per-kernel statistics say they are nothing alike.
            return (_centred(r) / r.std(dim=0, keepdim=True).clamp_min(1e-8)).flatten()
        if kernel >= r.shape[1]:
            raise NeurodesError(f"Kernel {kernel} does not exist: there are {r.shape[1]}.",
                                hint=f"Use 0 to {r.shape[1] - 1}, or -1 for all of them.")
        return r[:, kernel]

    after = pick(activation(model, layer, x))
    keep = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        model.reinitialise(seed=9000, only=layer)
        before = pick(activation(model, layer, x))
    finally:
        model.load_state_dict(keep)
    return after.cpu(), before.cpu()


@torch.no_grad()
def measure(model, data: DataBundle, limit: int = 4096, floor: bool = True,
            layer: str = "") -> dict:
    """Score the trained bank on held-out patches, against its own untrained floor."""
    model.eval()
    base = untrained_floor(model, data, limit=limit, layer=layer) if floor else None
    x = data.x_val if data.n_val >= 64 else data.x_train
    x = x[:limit].to(model.device)
    r = responses(activation(model, layer, x)).float()
    name, weight = weight_at(model, layer)
    overlap = kernel_overlap(weight) if weight is not None else None
    spread = r.std(dim=0)
    median = spread.median().clamp_min(1e-12)
    # Measured against whatever feeds this layer, because a bank can fall silent *together*
    # — the naive objective's favourite answer — and a purely relative test would call that
    # healthy. Against the layer's own input rather than the model's, so the number does not
    # simply accumulate with depth.
    below = feeds(model, layer)
    source = activation(model, below, x) if below else x
    loudness = (r.pow(2).mean().sqrt()
                / source.pow(2).mean().sqrt().clamp_min(1e-8)).item()
    return {
        "kernels": int(r.shape[1]),
        "samples": int(r.shape[0]),
        "loudness": loudness,
        "sparsity": sparsity(r).mean().item(),
        "kurtosis": kurtosis(r).mean().item(),
        "overlap": overlap.max(dim=1).values.mean().item() if overlap is not None else float("nan"),
        "worst_pair": overlap.max().item() if overlap is not None else float("nan"),
        # A kernel whose response barely moves has stopped saying anything about the image.
        "dead": int((spread < 0.2 * median).sum().item()),
        "layer": name,
        "floor": base,
    }


def report(model, data: DataBundle, diversity: float = 0.0, layer: str = "") -> str:
    m = measure(model, data, layer=layer)
    base = m["floor"]
    lines = [
        f"{m['kernels']} kernels scored on {m['samples']} held-out responses",
        "",
        f"  sparsity   E|r|/rms   {m['sparsity']:.3f}      "
        + (f"(untrained, this image: {base['sparsity']:.3f} — the number to beat)"
           if base else f"(gaussian {GAUSSIAN_L1L2:.3f})"),
        f"  peakiness  kurtosis   {m['kurtosis']:.2f}       "
        + (f"(untrained: {base['kurtosis']:.2f})" if base else
           f"(gaussian {GAUSSIAN_KURTOSIS:.2f})"),
    ]
    if not math.isnan(m["overlap"]):
        lines.append(f"  distinctness          {m['overlap']:.3f}      "
                     f"(1.000 would be every kernel identical; worst pair "
                     f"{m['worst_pair']:.3f})")
    lines.append(f"  dead kernels          {m['dead']} of {m['kernels']}")
    lines.append(f"  loudness              {m['loudness']:.3f}      "
                 f"(response size next to {'what feeds this layer' if layer else 'the input'})")
    lines.append("")
    lines += [f"  {line}" for line in verdict(m, diversity)]
    return "\n".join(lines)


def verdict(m: dict, diversity: float) -> list[str]:
    """Say the thing the numbers mean, since the loss curve will not say it."""
    out: list[str] = []
    base = m["floor"]
    learned = False
    if base:
        gained = base["sparsity"] - m["sparsity"]
        learned = gained > max(0.015, 3 * base["sparsity_sd"])
        # Anything inside a few times the re-initialisation spread is the picture talking,
        # not the training.
        if not learned:
            out.append(
                f"These kernels score {m['sparsity']:.3f} and untrained ones score "
                f"{base['sparsity']:.3f} on the same patches: nothing was learned. The score "
                "looks respectable only because photographs are non-Gaussian before any "
                "filter touches them.")
        elif gained > 0.06:
            out.append(f"Sparsity improved {gained:.3f} on the untrained floor — each kernel "
                       "is quiet over most of the image and loud in a few places, which is "
                       "what an edge detector does.")
        else:
            out.append(f"Sparsity improved only {gained:.3f} on the untrained floor. "
                       "Something was learned, but not much of what is probably there — "
                       "try more epochs, or the 'sparse response' objective if you are not "
                       "already using it.")
    if m["loudness"] < 0.05:
        out.append(f"The bank has switched itself off: its response is "
                   f"{1 / max(m['loudness'], 1e-9):.0f}x smaller than the input. An objective "
                   "that is not scale-invariant can be satisfied by outputting nothing, and "
                   "this one has been.")
    elif m["loudness"] > 10 and not learned:
        # Only a fault when it comes *instead of* structure. A layer partway up a stack is
        # legitimately several times louder than what feeds it, and saying so there would
        # be crying wolf at the healthy case.
        out.append(f"The response is {m['loudness']:.0f}x the size of what feeds it, and "
                   "nothing was learned. Gain is free to a filter and means nothing about "
                   "what it detects, so an objective that lets it grow this far is being "
                   "paid in volume rather than in structure.")
    if not math.isnan(m["overlap"]):
        if m["overlap"] > 0.7:
            out.append(f"The kernels have collapsed: on average each one is {m['overlap']:.0%} "
                       "the same as its nearest neighbour. "
                       + ("Raise diversity above 0." if diversity <= 0 else
                          "Raise diversity, or use fewer kernels than the patch can support."))
        elif m["overlap"] < 0.35:
            out.append("The kernels are genuinely different from one another.")
    if m["dead"]:
        out.append(f"{m['dead']} kernel(s) stopped responding to anything. Lower the "
                   "diversity weight, or train for longer.")
    return out or ["Nothing looks wrong."]
