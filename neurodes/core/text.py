"""Text, as something a network can be trained on — and read back afterwards.

A language model is a classifier with one peculiarity: it makes a prediction at **every
position**, and the answer at each position is simply the next character. That is the whole
of the training objective, and it needs no labelling at all — the text is its own answer,
shifted by one.

Characters rather than words or BPE tokens, deliberately. A character vocabulary is small
enough to train on a CPU, needs no tokenizer to install or explain, and makes the model's
progress legible: it learns spaces, then word shapes, then real words, then punctuation
that closes. Watching that happen in order is the point.
"""

from __future__ import annotations

import os

import torch

from .data import DataBundle
from .errors import NeurodesError
from .runtime import allocating

SAMPLE = """\
The wire carries a tensor and the node is a layer. Dropping a Linear onto a wire is the same
act as constructing a module and calling it. A tensor of one shape goes into a layer, a
tensor of another shape comes out, and much of what makes networks hard to learn is that
this is invisible until it crashes.

Shapes carry names, not just numbers. The batch dimension is written out on purpose, because
hiding it is where most shape confusion starts. Errors land on the node that caused them,
before any data exists, with a sentence you can act on.

Training changes the weights to suit the data. Deep dream holds the weights still and changes
the picture instead. Every position looks at every other position and pulls in what it finds
useful; the shape does not change, but each position now knows about its context.
"""


def _read(text: str, path: str, search: list[str]) -> tuple[str, str]:
    """Whichever of the two was given, plus a name for it."""
    if path:
        found = path if os.path.isfile(path) else ""
        for root in search:
            if found:
                break
            candidate = os.path.join(root, path)
            if os.path.isfile(candidate):
                found = candidate
        if not found:
            raise NeurodesError(
                f"No text file at {path!r}.",
                hint="Give a full path, or a name inside ComfyUI's input folder or this "
                     "pack's examples folder. Leave it empty to use the text typed on the "
                     "node instead.")
        with open(found, encoding="utf-8", errors="replace") as handle:
            return handle.read(), os.path.basename(found)
    return text, "typed text"


def text_dataset(text: str = "", path: str = "", context: int = 64, stride: int = 0,
                 val_fraction: float = 0.1, limit: int = 0,
                 search: list[str] | None = None) -> DataBundle:
    """Windows of characters, each paired with the same window shifted one to the left.

    ``context`` is how far back the model can see. ``stride`` is how far apart the windows
    start: 1 uses every position and makes the most of a small corpus, ``context`` uses each
    character once and is much faster. 0 picks something sensible in between.

    The split is taken by *position*, not at random — the tail of the text is held out whole.
    Shuffled windows would overlap across the split and the validation loss would be a lie.
    """
    body, name = _read(text or SAMPLE, path, search or [])
    if limit:
        body = body[:int(limit)]
    context = max(2, int(context))
    if len(body) < context + 2:
        raise NeurodesError(
            f"{name} has {len(body)} character(s), which is not enough for a context of "
            f"{context}.",
            hint="Use a longer text, or reduce 'context'.")

    vocabulary = sorted(set(body))
    index = {ch: i for i, ch in enumerate(vocabulary)}
    stride = max(1, int(stride) if stride else max(1, context // 4))

    with allocating():
        ids = torch.tensor([index[ch] for ch in body], dtype=torch.long)
        starts = torch.arange(0, len(ids) - context - 1, stride)
        if starts.numel() < 2:
            starts = torch.arange(0, max(1, len(ids) - context - 1))
        cut = max(1, int(starts.numel() * (1.0 - float(val_fraction))))
        # The held-out text is the tail, kept whole: windows overlap, so a random split
        # would put nearly identical sequences on both sides and flatter the score.
        def windows(chosen):
            x = torch.stack([ids[s: s + context] for s in chosen.tolist()])
            y = torch.stack([ids[s + 1: s + context + 1] for s in chosen.tolist()])
            return x, y

        x_train, y_train = windows(starts[:cut])
        x_val, y_val = windows(starts[cut:] if starts.numel() > cut else starts[-1:])

    return DataBundle(
        x_train, y_train, x_val, y_val, task="classification",
        classes=tuple(vocabulary), name=f"{name} ({len(body):,} characters)",
        notes=f"{len(vocabulary)} distinct character(s), {context} of context. The answer at "
              "every position is the next character, so nothing was labelled.",
        extra={"text": {"vocabulary": vocabulary, "context": context, "source": name}},
    )


def config_of(bundle: DataBundle) -> dict:
    """The vocabulary a dataset was built with, or a clear complaint."""
    cfg = (bundle.extra or {}).get("text")
    if not cfg:
        raise NeurodesError(
            f"{bundle.name} is not a text dataset.",
            hint="Connect the Text Dataset node that trained this model. It carries the "
                 "vocabulary, which is the only way to turn numbers back into letters.")
    return cfg


def encode(body: str, vocabulary: list[str]) -> torch.Tensor:
    index = {ch: i for i, ch in enumerate(vocabulary)}
    return torch.tensor([index[ch] for ch in body if ch in index], dtype=torch.long)


def decode(ids: torch.Tensor, vocabulary: list[str]) -> str:
    return "".join(vocabulary[int(i)] for i in ids.reshape(-1).tolist())


@torch.no_grad()
def generate(model, cfg: dict, prompt: str = "", length: int = 400,
             temperature: float = 0.8, top_k: int = 0, seed: int = 0) -> str:
    """Write text one character at a time, feeding each guess back in.

    The loop is the same one behind every language model: ask for the distribution over the
    next character, draw one from it, append it, and ask again. ``temperature`` is how
    boldly to draw — near 0 it always takes the most likely character and gets stuck in
    loops; above about 1.2 it produces noise.
    """
    vocabulary = list(cfg["vocabulary"])
    context = int(cfg.get("context", 64))
    device = getattr(model, "device", None) or torch.device("cpu")
    model.eval()
    g = torch.Generator().manual_seed(int(seed))

    seed_ids = encode(prompt, vocabulary)
    if seed_ids.numel() == 0:
        seed_ids = torch.randint(0, len(vocabulary), (1,), generator=g)
    ids = seed_ids.clone()

    for _ in range(max(1, int(length))):
        window = ids[-context:]
        if window.numel() < context:                       # left-pad the first few steps
            window = torch.cat([window.new_zeros(context - window.numel()), window])
        logits = model(window.unsqueeze(0).to(device))
        step = logits[0, -1] if logits.dim() == 3 else logits[0]
        step = step.float().cpu() / max(1e-3, float(temperature))
        if top_k:
            k = min(int(top_k), step.numel())
            cut = step.topk(k).values[-1]
            step = step.masked_fill(step < cut, float("-inf"))
        nxt = torch.multinomial(torch.softmax(step, dim=-1), 1, generator=g)
        ids = torch.cat([ids, nxt])

    return decode(ids, vocabulary)
