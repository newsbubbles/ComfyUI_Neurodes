"""Composite ``nn.Module`` classes used by the block layers.

These are written as ordinary, readable PyTorch on purpose: the code exporter lifts them
out of this file with ``inspect.getsource`` and pastes them into the file it generates. So
what you read here is exactly what a user gets when they export their workflow, and the
exported model cannot drift from the one that trained.

Keep them dependency-free (torch only) and keep them readable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

ACTS = {
    "relu": F.relu, "gelu": F.gelu, "silu": F.silu, "tanh": torch.tanh,
    "leaky_relu": F.leaky_relu, "elu": F.elu, "mish": F.mish,
    "sigmoid": torch.sigmoid, "none": lambda t: t,
}


def causal_mask(length: int, device) -> torch.Tensor:
    """True where a position must not look, i.e. strictly above the diagonal."""
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


class MLPBlock(nn.Module):
    """A run of Linear layers with an activation, optional norm and dropout between them."""

    def __init__(self, in_features: int, hidden: int, out_features: int, depth: int = 2,
                 activation: str = "relu", dropout: float = 0.0, norm: bool = False):
        super().__init__()
        self.act = activation
        widths = [in_features] + [hidden] * max(depth - 1, 0) + [out_features]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.norms = nn.ModuleList(
            nn.LayerNorm(w) if norm else nn.Identity() for w in widths[1:-1]
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != last:                      # no activation on the output layer
                x = self.norms[i](x)
                x = ACTS[self.act](x)
                x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    """Convolution, then normalisation, then activation, optionally halving the size."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, norm: str = "batch", activation: str = "relu",
                 pool: bool = False, groups: int = 1):
        super().__init__()
        self.act = activation
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=(norm == "none"))
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "group":
            self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        else:
            self.norm = nn.Identity()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(ACTS[self.act](self.norm(self.conv(x))))


class ResidualBlock(nn.Module):
    """Two convolutions with the input added back on — the ResNet building block.

    The shortcut is a 1x1 convolution only when the shape changes, otherwise it is the
    identity, which is the whole point: the gradient has an unobstructed path back.
    """

    def __init__(self, channels: int, out_channels: int = 0, stride: int = 1,
                 activation: str = "relu", norm: str = "batch", skip: bool = True):
        super().__init__()
        out_channels = out_channels or channels
        self.act = activation
        self.skip = skip
        self.conv1 = nn.Conv2d(channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        make_norm = (lambda c: nn.BatchNorm2d(c)) if norm == "batch" else \
                    (lambda c: nn.GroupNorm(min(8, c), c)) if norm == "group" else \
                    (lambda c: nn.Identity())
        self.norm1, self.norm2 = make_norm(out_channels), make_norm(out_channels)
        # With the skip switched off there is nothing to add the shortcut to, so the
        # projection is not built at all. That is why a plain stack has *fewer* weights
        # than a residual one of the same depth, and the difference is exactly the
        # projections at the two places the channel count changes.
        if skip and (stride != 1 or out_channels != channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, out_channels, 1, stride=stride, bias=False),
                make_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = ACTS[self.act](self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if not self.skip:
            return ACTS[self.act](h)
        return ACTS[self.act](h + self.shortcut(x))


class TransformerBlock(nn.Module):
    """Pre-norm self-attention plus a feed-forward, each wrapped in a residual.

    Pre-norm (normalise going in, rather than coming out) is used because it trains without
    a learning-rate warmup, which matters a lot when someone is experimenting.
    """

    def __init__(self, d_model: int, num_heads: int = 4, ff_mult: float = 4.0,
                 dropout: float = 0.0, causal: bool = False, activation: str = "gelu"):
        super().__init__()
        self.causal = causal
        self.act = activation
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = max(1, int(d_model * ff_mult))
        self.ff1 = nn.Linear(d_model, hidden)
        self.ff2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        mask = causal_mask(x.shape[1], x.device) if self.causal else None
        attended, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(attended)
        h = self.norm2(x)
        return x + self.drop(self.ff2(ACTS[self.act](self.ff1(h))))


class TransformerStack(nn.Module):
    """``repeat`` transformer blocks in a row. Each one gets its own weights."""

    def __init__(self, d_model: int, num_heads: int = 4, ff_mult: float = 4.0,
                 dropout: float = 0.0, causal: bool = False, activation: str = "gelu",
                 repeat: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, num_heads, ff_mult, dropout, causal, activation)
            for _ in range(repeat)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SinusoidalPositions(nn.Module):
    """Adds fixed sine and cosine waves so the model can tell positions apart.

    Attention alone is order-blind: shuffle the sequence and the output is shuffled the same
    way. This is the classic fix and it learns nothing.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]].to(x.dtype)


class LearnedPositions(nn.Module):
    """A position embedding the network trains for itself."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        idx = torch.arange(x.shape[1], device=x.device)
        return x + self.pos(idx).unsqueeze(0)
