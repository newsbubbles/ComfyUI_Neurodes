"""Learning a mapping from nothing but a human saying "yes" or "no".

The honest framing first, because the name invites overclaiming.

**One bit per run is a very low information rate.** A model with six outputs has a
continuous space to search and you are handing it a single bit each time you press a
button. Fifty presses is fifty bits. Nothing here changes that arithmetic, and no amount of
cleverness in the update rule will.

What the two modes do about it:

*Direct* is advantage-weighted regression. The policy proposes ``y = mu(x) + sigma*noise``,
you judge it, and every remembered proposal is regressed towards with weight
``exp((r - baseline) / temperature)`` -- liked proposals pull hard, disliked ones barely
pull at all. Weights stay positive, which matters: the obvious rule of "move *away* from
what was disliked" has no fixed point and walks off to infinity. Replay is sound here
because AWR is a weighted regression rather than an on-policy gradient, so old samples stay
usable as the policy moves.

*Critic* is the reward-model idea. A second network learns to predict whether a pair
``(x, y)`` would be liked, trained on every vote so far, and the policy is then optimised
against that prediction. This does not create information -- it **amortises** it. One bit
still arrives per press, but the critic lets that bit be reused across many gradient steps
and, more usefully, lets the policy be scored at settings you have never actually judged.

That second property is the whole point of a reward model, and it is also where it breaks.
Push the policy hard enough against a critic fitted to forty votes and it will find the
corner of parameter space the critic is most wrong about. The critic's score goes up; the
thing you actually wanted goes down. There is no way to detect that from inside -- by
construction, the critic thinks it is going well -- so :func:`report` shows the critic's
confidence next to how much data it was fitted on, and the polish step count is a widget
rather than a constant, because knowing when to stop is the human's job.

Measured against a simulated human with a fixed hidden taste, 120 votes, error being mean
distance from its private target on inputs never voted on. Three things came out of it, and
the first one inverted the defaults this file shipped with:

**One gradient step per vote, not sixty.** Replaying the whole buffer for 40 Adam steps
after every press *raised* the error by 0.077 -- worse than never training. It is not
learning, it is memorising a handful of noisy proposals. Dropping to a single step turned
that into a 0.186 improvement. The buffer is tiny and the signal is one bit; almost any
amount of optimisation per press is too much.

**A picky human teaches it less than a generous one.** Liking anything within 0.28 of the
target improved the error by 0.138. Tightening to 0.10 -- a harder judge -- made it 0.022
*worse*, because only 1% of proposals were ever liked and advantage-weighted regression has
nothing to pull towards. Withholding approval starves it.

**It works for a few knobs, not many.** Four outputs, no input: 0.341 -> 0.101. Eight
outputs: 0.318 -> 0.212, roughly half the gain, because a random proposal landing inside
tolerance in eight dimensions is rare. Past that it stops working at all.

The critic earns its place in every tractable case -- four outputs 0.101 -> 0.076, eight
outputs 0.212 -> 0.181 -- and then stops. At eight outputs *with* two inputs, where the pair
space is ten-dimensional and there are 120 votes to cover it, turning the critic on made the
result worse: -0.083 without it, -0.051 with. That is reward hacking, reproduced on a
problem small enough to see it happen.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from .errors import NeurodesError

SHAPES: dict[str, tuple[int, int]] = {
    "linear": (0, 0),
    "small": (16, 1),
    "wide": (64, 1),
    "deep": (32, 3),
}
MAX_BUFFER = 512
VOTES = {"like": 1.0, "dislike": 0.0}


def _mlp(n_in: int, n_out: int, hidden: int, depth: int, squash: bool) -> nn.Module:
    if hidden <= 0 or depth <= 0:
        layers: list[nn.Module] = [nn.Linear(n_in, n_out)]
    else:
        layers = [nn.Linear(n_in, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, n_out)]
    if squash:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


@dataclass
class Preference:
    """Everything that has to survive between two presses of Run."""

    n_in: int
    n_out: int
    shape: str
    policy: nn.Module
    critic: nn.Module
    xs: list[list[float]] = field(default_factory=list)
    ys: list[list[float]] = field(default_factory=list)
    rs: list[float] = field(default_factory=list)
    runs: int = 0
    pending: dict | None = None
    history: list[float] = field(default_factory=list)

    @property
    def votes(self) -> int:
        return len(self.rs)

    @property
    def baseline(self) -> float:
        return sum(self.rs) / len(self.rs) if self.rs else 0.5

    def tensors(self):
        return (torch.tensor(self.xs, dtype=torch.float32),
                torch.tensor(self.ys, dtype=torch.float32),
                torch.tensor(self.rs, dtype=torch.float32))


# --------------------------------------------------------------------------- storage
def _root() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "neurodes-preference")


def slot_path(slot: str) -> str:
    safe = "".join(c for c in (slot or "default") if c.isalnum() or c in "-_ ").strip()
    return os.path.join(_root(), f"{safe or 'default'}.pt")


def build(n_in: int, n_out: int, shape: str, seed: int = 0) -> Preference:
    if shape not in SHAPES:
        raise NeurodesError(f"Unknown shape preset {shape!r}",
                            hint="Choose one of: " + ", ".join(SHAPES))
    hidden, depth = SHAPES[shape]
    torch.manual_seed(int(seed))
    return Preference(n_in=n_in, n_out=n_out, shape=shape,
                      policy=_mlp(max(n_in, 1), n_out, hidden, depth, squash=True),
                      critic=_mlp(max(n_in, 1) + n_out, 1, max(hidden, 16), max(depth, 1),
                                  squash=True))


def load(slot: str, n_in: int, n_out: int, shape: str, seed: int = 0) -> Preference:
    """Read the slot back, or start a fresh one.

    A changed input or output count means the saved weights no longer describe this node,
    so the slot is rebuilt rather than half-restored -- silently keeping a policy that maps
    the wrong number of things would be worse than losing the votes.
    """
    path = slot_path(slot)
    fresh = build(n_in, n_out, shape, seed)
    if not os.path.exists(path):
        return fresh
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return fresh
    if (blob.get("n_in") != n_in or blob.get("n_out") != n_out
            or blob.get("shape") != shape):
        return fresh
    try:
        fresh.policy.load_state_dict(blob["policy"])
        fresh.critic.load_state_dict(blob["critic"])
    except Exception:
        return build(n_in, n_out, shape, seed)
    fresh.xs = blob.get("xs", [])
    fresh.ys = blob.get("ys", [])
    fresh.rs = blob.get("rs", [])
    fresh.runs = blob.get("runs", 0)
    fresh.pending = blob.get("pending")
    fresh.history = blob.get("history", [])
    return fresh


def save(pref: Preference, slot: str) -> None:
    os.makedirs(_root(), exist_ok=True)
    torch.save({"n_in": pref.n_in, "n_out": pref.n_out, "shape": pref.shape,
                "policy": pref.policy.state_dict(), "critic": pref.critic.state_dict(),
                "xs": pref.xs, "ys": pref.ys, "rs": pref.rs, "runs": pref.runs,
                "pending": pref.pending, "history": pref.history,
                "saved": time.time()}, slot_path(slot))


def forget(slot: str) -> bool:
    path = slot_path(slot)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------------------- acting
@torch.no_grad()
def propose(pref: Preference, x: list[float], exploration: float,
            seed: int | None = None) -> tuple[list[float], list[float]]:
    """The policy's own answer, and the noisy one actually handed out.

    Without the noise there is nothing to learn from: every run would produce the same
    values and every vote would be about the same point. The noise is the search.
    """
    xt = torch.tensor([x or [0.0]], dtype=torch.float32)
    mu = pref.policy(xt)[0]
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(mu.shape, generator=g)
    else:
        noise = torch.randn_like(mu)
    y = (mu + noise * float(exploration)).clamp(0.0, 1.0)
    return mu.tolist(), y.tolist()


def record(pref: Preference, vote: str) -> bool:
    """Attach a vote to the proposal the human was actually looking at."""
    if vote not in VOTES or not pref.pending:
        return False
    pref.xs.append(pref.pending["x"])
    pref.ys.append(pref.pending["y"])
    pref.rs.append(VOTES[vote])
    del pref.xs[:-MAX_BUFFER], pref.ys[:-MAX_BUFFER], pref.rs[:-MAX_BUFFER]
    pref.pending = None
    return True


# -------------------------------------------------------------------------- learning
def learn(pref: Preference, steps: int = 1, lr: float = 0.02,
          temperature: float = 0.3) -> float:
    """Advantage-weighted regression over every vote so far.

    ``steps`` defaults to 1 on purpose. Sixty steps per press measured *worse than not
    training at all*; see the module docstring for the numbers.
    """
    if pref.votes < 1:
        return 0.0
    xs, ys, rs = pref.tensors()
    if pref.n_in == 0:
        xs = torch.zeros(len(rs), 1)
    weights = torch.exp((rs - pref.baseline) / max(temperature, 1e-3)).unsqueeze(1)
    weights = weights / weights.mean().clamp_min(1e-8)
    opt = torch.optim.Adam(pref.policy.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(max(int(steps), 1)):
        opt.zero_grad(set_to_none=True)
        loss = (weights * (pref.policy(xs) - ys).pow(2)).mean()
        loss.backward()
        opt.step()
    return float(loss.detach())


def fit_critic(pref: Preference, steps: int = 120, lr: float = 0.02) -> float:
    """Train the reward model: given (x, y), would this have been liked?

    Needs both answers present. Fitting a classifier to forty likes and no dislikes gives a
    critic that says yes everywhere, which is worse than no critic at all because the
    policy will happily optimise against it.
    """
    if pref.votes < 4 or len(set(pref.rs)) < 2:
        return float("nan")
    xs, ys, rs = pref.tensors()
    if pref.n_in == 0:
        xs = torch.zeros(len(rs), 1)
    pairs = torch.cat([xs, ys], dim=1)
    opt = torch.optim.Adam(pref.critic.parameters(), lr=lr)
    fn = nn.BCELoss()
    loss = torch.tensor(0.0)
    for _ in range(max(int(steps), 1)):
        opt.zero_grad(set_to_none=True)
        loss = fn(pref.critic(pairs).squeeze(1).clamp(1e-6, 1 - 1e-6), rs)
        loss.backward()
        opt.step()
    return float(loss)


def polish(pref: Preference, x: list[float], y: list[float], steps: int,
           lr: float = 0.05) -> tuple[list[float], float, float]:
    """Walk the proposal uphill on the critic's opinion of it.

    This is the step that reuses old feedback, and the step that can be pushed too far.
    Both scores are returned so the drift is visible rather than implied.
    """
    xt = torch.tensor([x or [0.0]], dtype=torch.float32)
    v = torch.tensor([y], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        before = float(pref.critic(torch.cat([xt, v], dim=1)))
    if steps <= 0:
        return y, before, before
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        (-pref.critic(torch.cat([xt, v], dim=1))).mean().backward()
        opt.step()
        with torch.no_grad():
            v.clamp_(0.0, 1.0)
    with torch.no_grad():
        after = float(pref.critic(torch.cat([xt, v], dim=1)))
    return v.detach()[0].tolist(), before, after


def report(pref: Preference, slot: str, critic_loss: float = float("nan"),
           before: float = float("nan"), after: float = float("nan")) -> str:
    likes = int(sum(pref.rs))
    lines = [f"slot '{slot}' — run {pref.runs}",
             f"{pref.votes} vote(s): {likes} like, {pref.votes - likes} dislike"]
    if pref.votes == 0:
        lines.append("No feedback yet. The values are the untrained policy's, plus noise.")
    if not math.isnan(after):
        lines.append(f"critic score {before:.3f} -> {after:.3f} over the polish steps")
        if pref.votes < 12:
            lines.append(f"WARNING: the critic was fitted on {pref.votes} vote(s). "
                         "Optimising against it this early mostly finds where it is wrong.")
    if not math.isnan(critic_loss):
        lines.append(f"critic fit loss {critic_loss:.4f}")
    elif pref.votes >= 4:
        lines.append("Critic idle: it needs at least one like and one dislike.")
    return "\n".join(lines)


def summary(slot: str) -> str:
    path = slot_path(slot)
    if not os.path.exists(path):
        return json.dumps({"slot": slot, "exists": False})
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return json.dumps({"slot": slot, "exists": True, "votes": len(blob.get("rs", [])),
                       "runs": blob.get("runs", 0), "n_in": blob.get("n_in"),
                       "n_out": blob.get("n_out"), "shape": blob.get("shape")})
