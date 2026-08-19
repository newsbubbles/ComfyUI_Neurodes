"""Training: the button that turns an architecture into a model.

The loop itself is deliberately plain — the interesting part is what happens *before* it.
:func:`check_compatibility` catches the handful of mistakes that account for most failed
first attempts (wrong number of outputs, a softmax in front of a loss that applies its own,
integer targets fed to a regression loss) and explains them in a sentence, before any time
is spent training.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from .compile import CompiledModel
from .data import DataBundle
from .errors import NeurodesError
from .runtime import trainable

OPTIMIZERS = ("adam", "adamw", "sgd", "sgd + momentum", "rmsprop", "adagrad", "adadelta")
LOSSES = ("auto", "cross entropy", "mse", "mae", "huber", "binary cross entropy", "nll")

_LOSS_FNS = {
    "cross entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
    "binary cross entropy": nn.BCEWithLogitsLoss,
    "nll": nn.NLLLoss,
}

_LOSS_SRC = {
    "cross entropy": "nn.CrossEntropyLoss()",
    "mse": "nn.MSELoss()",
    "mae": "nn.L1Loss()",
    "huber": "nn.SmoothL1Loss()",
    "binary cross entropy": "nn.BCEWithLogitsLoss()",
    "nll": "nn.NLLLoss()",
}


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "auto"
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    device: str = "auto"
    seed: int = 0
    shuffle: bool = True
    early_stopping: int = 20
    """Stop after this many epochs with no improvement in validation loss. 0 disables it.

    With this on, ``epochs`` becomes a ceiling rather than a prescription, which is the
    more useful way to think about it: set it generously and let the run decide when it is
    finished. The best weights are restored, so stopping late costs nothing.
    """


@dataclass
class History:
    """Everything worth plotting afterwards."""

    step_loss: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    task: str = "classification"
    loss_name: str = ""
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    stopped_early: bool = False
    best_epoch: int = 0
    """The epoch whose weights the model ended up with."""

    @property
    def best_val_acc(self) -> float:
        return max(self.val_acc) if self.val_acc else float("nan")

    @property
    def final_val_loss(self) -> float:
        return self.val_loss[-1] if self.val_loss else float("nan")

    @property
    def restored(self) -> bool:
        """Whether the model ended up with earlier weights than the last epoch's."""
        return bool(self.best_epoch) and self.best_epoch != len(self.epochs)

    @property
    def kept(self) -> int:
        """Index into the per-epoch lists of the weights the model actually has."""
        return self.best_epoch - 1 if self.restored else -1

    def summary(self) -> str:
        # Every "->" below reports the epoch whose weights the model is *holding*, which
        # after a restore is not the last one. Reporting the last epoch's numbers there
        # would describe a model that was thrown away.
        kept = self.kept
        lines = [f"trained for {len(self.epochs)} epoch(s) in {self.seconds:.1f}s"
                 + ("  (stopped early)" if self.stopped_early else "")
                 + (f", kept epoch {self.best_epoch}" if self.restored else "")]
        if self.train_loss:
            lines.append(f"  loss        {self.train_loss[0]:.4f}  ->  {self.train_loss[kept]:.4f}")
        if self.val_loss:
            lines.append(f"  val loss    {self.val_loss[0]:.4f}  ->  {self.val_loss[kept]:.4f}")
        if self.task == "classification" and self.val_acc:
            lines.append(f"  val acc     {self.val_acc[0] * 100:.1f}%  ->  "
                         f"{self.val_acc[kept] * 100:.1f}%"
                         f"   (best {self.best_val_acc * 100:.1f}%)")
        if self.notes:
            lines.append("")
            lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_loss(name: str, data: DataBundle) -> str:
    if name != "auto":
        return name
    return "cross entropy" if data.task == "classification" else "mse"


def _shape_mismatch(declared, actual: list[int]) -> bool:
    want = list(declared.dims)[1:]
    return len(want) != len(actual) or any(
        d.is_concrete and d.size != a for d, a in zip(want, actual))


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    table = {
        "adam": lambda: torch.optim.Adam(params, lr=lr, weight_decay=weight_decay),
        "adamw": lambda: torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay or 0.01),
        "sgd": lambda: torch.optim.SGD(params, lr=lr, weight_decay=weight_decay),
        "sgd + momentum": lambda: torch.optim.SGD(params, lr=lr, momentum=0.9,
                                                  weight_decay=weight_decay),
        "rmsprop": lambda: torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay),
        "adagrad": lambda: torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay),
        "adadelta": lambda: torch.optim.Adadelta(params, lr=lr, weight_decay=weight_decay),
    }
    if name not in table:
        raise NeurodesError(f"Unknown optimizer {name!r}", hint="Choose one of: " + ", ".join(table))
    return table[name]()


def optimizer_src(name: str, lr: float, weight_decay: float) -> str:
    body = {
        "adam": f"torch.optim.Adam(model.parameters(), lr={lr}",
        "adamw": f"torch.optim.AdamW(model.parameters(), lr={lr}",
        "sgd": f"torch.optim.SGD(model.parameters(), lr={lr}",
        "sgd + momentum": f"torch.optim.SGD(model.parameters(), lr={lr}, momentum=0.9",
        "rmsprop": f"torch.optim.RMSprop(model.parameters(), lr={lr}",
        "adagrad": f"torch.optim.Adagrad(model.parameters(), lr={lr}",
        "adadelta": f"torch.optim.Adadelta(model.parameters(), lr={lr}",
    }.get(name, f"torch.optim.Adam(model.parameters(), lr={lr}")
    return body + (f", weight_decay={weight_decay})" if weight_decay else ")")


def loss_src(name: str) -> str:
    return _LOSS_SRC.get(name, "nn.CrossEntropyLoss()")


# ---------------------------------------------------------------------------
# The checks that save an hour
# ---------------------------------------------------------------------------

def check_compatibility(model: CompiledModel, data: DataBundle, loss_name: str) -> list[str]:
    """Raise on anything that cannot work; return a note for anything merely suspect."""
    notes: list[str] = []

    if len(model.input_ops) != data.n_inputs:
        names = ", ".join(op.params.get("name", "?") for op in model.input_ops)
        raise NeurodesError(
            f"This model takes {len(model.input_ops)} input(s) ({names}), but the dataset "
            f"supplies {data.n_inputs}.",
            hint=("Add an Input node for each one and wire the dataset's extra shape outputs "
                  "into them." if len(model.input_ops) < data.n_inputs else
                  "Remove the spare Input node, or use a dataset that provides the other "
                  "input — Dataset As Diffusion with `timestep` set to 'second input' is an "
                  "example of one that does."),
        )

    actual = list(data.x_train.shape[1:])
    for i, (declared, given) in enumerate(zip(model.input_shapes, data.input_shapes)):
        want = [int(d.size) for d in given.dims[1:]]
        if _shape_mismatch(declared, want):
            which = "The Input node" if data.n_inputs == 1 else f"Input {i + 1}"
            raise NeurodesError(
                f"{which} says {declared}, but that input's data is "
                f"[{data.n_train}, {', '.join(str(a) for a in want)}].",
                hint=f"Set its shape to 'B, {', '.join(str(a) for a in want)}', or use the "
                     "Dataset node's shape output to drive it directly.",
            )

    out_shape = model.output_shapes[0]
    last_op = model.plan[-1].op if model.plan else None
    last_kind = last_op.kind if last_op else ""

    if data.task == "classification":
        # A language model predicts a class at every *position*, so the target has a time
        # axis too. Say so plainly when the graph has pooled that axis away, because the
        # error torch would give instead is about tensor sizes and names nothing.
        if data.y_train.dim() == 2 and out_shape.rank != data.y_train.dim() + 1:
            raise NeurodesError(
                f"Each target is a sequence of {data.y_train.shape[1]} label(s), so the model "
                f"has to answer at every position — but it returns {out_shape}.",
                hint="Keep the time axis all the way to the end: a Linear applied to "
                     f"[B, T, features] gives [B, T, {data.n_classes}], which is one "
                     "prediction per position. A Reduce or a Global Pool in the middle "
                     "collapses it.",
            )
        width = out_shape[-1]
        if width.is_concrete and width.size != data.n_classes:
            if width.size == 1 and data.n_classes == 2:
                notes.append("A single output for two classes only works with binary cross "
                             "entropy. Use 2 outputs with cross entropy instead if unsure.")
            else:
                raise NeurodesError(
                    f"The model ends with {width} output(s) but the dataset has "
                    f"{data.n_classes} classes.",
                    hint=f"Set the final Linear layer's units to {data.n_classes}. A classifier "
                         "produces one score per class.",
                )
        if loss_name == "cross entropy" and last_kind in ("softmax", "log_softmax"):
            notes.append(
                f"There is a {last_kind.replace('_', ' ')} on the output and cross entropy "
                "applies its own. Training will still run but will be weaker — remove the "
                "activation, or switch the loss to NLL.")
        if loss_name == "nll" and last_kind != "log_softmax":
            notes.append("NLL loss expects log-probabilities. Put a Log Softmax on the output.")
        if loss_name in ("mse", "mae", "huber"):
            raise NeurodesError(
                f"{loss_name} is a regression loss, but this is a classification dataset with "
                "whole-number labels.",
                hint="Use cross entropy, or set the loss to 'auto'.",
            )
    elif data.task == "reconstruction":
        # The model has to produce whatever the target is. For an autoencoder that happens
        # to be the input, but for a denoiser or a colouriser it is a different picture,
        # and for colourisation a different channel count, so compare against the target.
        target = list(data.y_train.shape[1:])
        if _shape_mismatch(out_shape, target):
            same = target == actual
            raise NeurodesError(
                f"This model returns {out_shape}, but each target is "
                f"[{', '.join(str(t) for t in target)}].",
                hint=("The decoder must undo the encoder exactly. For images that usually "
                      "means matching each stride-2 convolution with an Upsample or a "
                      "Conv Transpose 2D, and finishing on the same channel count."
                      if same else
                      f"The input is [{', '.join(str(a) for a in actual)}] and the target is "
                      f"[{', '.join(str(t) for t in target)}], so the last layer needs "
                      f"{target[0]} output channel(s)."),
            )
        if loss_name in ("cross entropy", "nll"):
            raise NeurodesError(
                f"{loss_name} is a classification loss, but an autoencoder is predicting "
                "values, not classes.",
                hint="Use mse, mae or huber, or set the loss to 'auto'.",
            )
        # Only worth saying when the target really is a picture. A diffusion model predicts
        # noise, which is centred on zero and runs negative, and a Sigmoid on the end of one
        # would make it unable to produce half its answers.
        bounded_target = bool(data.y_train.numel()) and float(data.y_train.min()) >= -1e-4
        if last_kind not in ("sigmoid", "tanh", "clamp") and bounded_target:
            notes.append(
                "The output is unbounded, while the target is between 0 and 1. A Sigmoid on "
                "the end usually converges faster and stops the model wasting capacity "
                "learning to stay in range.")
    else:
        target_width = data.y_train.shape[1] if data.y_train.dim() > 1 else 1
        width = out_shape[-1]
        if width.is_concrete and width.size != target_width:
            raise NeurodesError(
                f"The model produces {width} value(s) per example but each target has "
                f"{target_width}.",
                hint=f"Set the final Linear layer's units to {target_width}.",
            )
        if loss_name in ("cross entropy", "nll"):
            raise NeurodesError(
                f"{loss_name} is a classification loss, but this dataset has continuous targets.",
                hint="Use mse, mae or huber — or set the loss to 'auto'.",
            )
    return notes


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.shape[-1] == 1:
        predicted = (logits.squeeze(-1) > 0).long()
    else:
        predicted = logits.argmax(dim=-1)
    return (predicted == targets.reshape(predicted.shape)).float().mean().item()


def _prepare_targets(y: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name in ("cross entropy", "nll"):
        return y.long()
    if loss_name == "binary cross entropy":
        return y.float()
    return y.float()


def _flatten_sequence(logits: torch.Tensor, target: torch.Tensor):
    """One prediction per *position*, folded down to one prediction per row.

    A language model produces ``[batch, time, vocabulary]`` and is scored against
    ``[batch, time]`` — every position in every sequence is its own classification. Torch's
    cross entropy wants two dimensions, so the batch and time axes are folded together and
    the loss is the mean over all of them, which is exactly what it should be.

    Only touches the case it is for: a rank-3 output against a rank-2 whole-number target.
    """
    if logits.dim() >= 3 and target.dim() == logits.dim() - 1:
        return logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
    return logits, target


@torch.no_grad()
def evaluate(model: CompiledModel, x, y: torch.Tensor, loss_fn,
             loss_name: str, task: str, batch_size: int = 512) -> tuple[float, float]:
    """Mean loss and (for classification) accuracy over a whole split.

    ``x`` is one tensor, or a sequence of them for a model that takes several inputs.
    Batches are moved to wherever the weights are, so this can be called with CPU tensors
    against a model that training left on the GPU.
    """
    model.eval()
    xs = tuple(x) if isinstance(x, (list, tuple)) else (x,)
    device = getattr(model, "device", None) or next(model.parameters()).device
    total_loss, total_acc, seen = 0.0, 0.0, 0
    for start in range(0, xs[0].shape[0], batch_size):
        batch = [t[start:start + batch_size].to(device) for t in xs]
        xb = batch[0]
        yb = y[start:start + batch_size].to(device)
        if xb.shape[0] == 0:
            continue
        logits = model(*batch)
        target = _prepare_targets(yb, loss_name)
        if loss_name in ("cross entropy", "nll"):
            logits, target = _flatten_sequence(logits, target)
        if loss_name == "binary cross entropy" and logits.shape != target.shape:
            target = target.view_as(logits)
        loss = loss_fn(logits, target)
        n = xb.shape[0]
        total_loss += loss.item() * n
        if task == "classification":
            total_acc += _accuracy(logits, target) * n
        seen += n
    if seen == 0:
        return float("nan"), float("nan")
    return total_loss / seen, (total_acc / seen if task == "classification" else float("nan"))


def train(model: CompiledModel, data: DataBundle, cfg: TrainConfig,
          on_progress: Callable[[int, int, dict], None] | None = None,
          should_stop: Callable[[], bool] | None = None,
          history: History | None = None) -> History:
    """Fit ``model`` to ``data``. Returns the history; the model is modified in place.

    Pass ``history`` to have the results written into an object you already hold, so a
    caller can watch the curve fill up while training is still running.
    """
    # Weights, data and gradients all have to exist outside the host's inference mode.
    with trainable():
        return _fit(model, data, cfg, on_progress, should_stop, history)


def _fit(model: CompiledModel, data: DataBundle, cfg: TrainConfig,
         on_progress, should_stop, history: History | None) -> History:
    torch.manual_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    loss_name = resolve_loss(cfg.loss, data)
    notes = check_compatibility(model, data, loss_name)

    model.to(device)
    bundle = data.to(device)
    loss_fn = _LOSS_FNS[loss_name]()
    optimizer = make_optimizer(cfg.optimizer, model.parameters(), float(cfg.learning_rate),
                               float(cfg.weight_decay))
    if not list(model.parameters()):
        raise NeurodesError(
            "This model has no trainable weights, so there is nothing to train.",
            hint="Add at least one layer that has parameters, such as Linear or Conv 2D.",
        )

    history = history if history is not None else History()
    history.task, history.loss_name = bundle.task, loss_name
    history.notes.extend(notes)
    n = bundle.n_train
    batch = max(1, min(int(cfg.batch_size), n))
    steps_per_epoch = max(1, math.ceil(n / batch))
    epochs = max(1, int(cfg.epochs))
    total_steps = steps_per_epoch * epochs
    started = time.time()
    step = 0
    patience = max(0, int(cfg.early_stopping))
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, device=device) if cfg.shuffle else torch.arange(n, device=device)
        running, seen = 0.0, 0
        for start in range(0, n, batch):
            if should_stop is not None and should_stop():
                history.stopped_early = True
                break
            idx = order[start:start + batch]
            inputs = [t[idx] for t in bundle.train_inputs]
            xb, yb = inputs[0], bundle.y_train[idx]
            logits = model(*inputs)
            target = _prepare_targets(yb, loss_name)
            if loss_name in ("cross entropy", "nll"):
                logits, target = _flatten_sequence(logits, target)
            if loss_name == "binary cross entropy" and logits.shape != target.shape:
                target = target.view_as(logits)
            loss = loss_fn(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

            value = loss.item()
            if not math.isfinite(value):
                history.notes.append(
                    f"The loss became {value} at epoch {epoch + 1}. Training stopped. "
                    "This almost always means the learning rate is too high.")
                history.stopped_early = True
                break
            history.step_loss.append(value)
            running += value * xb.shape[0]
            seen += xb.shape[0]
            step += 1
            if on_progress is not None:
                on_progress(step, total_steps, {"epoch": epoch + 1, "loss": value})

        if seen:
            train_loss = running / seen
            _, train_acc = evaluate(model, bundle.train_inputs, bundle.y_train, loss_fn,
                                    loss_name, bundle.task)
            val_loss, val_acc = evaluate(model, bundle.val_inputs, bundle.y_val, loss_fn,
                                         loss_name, bundle.task)
            history.epochs.append(epoch + 1)
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.train_acc.append(train_acc)
            history.val_acc.append(val_acc)

            if patience:
                # Improvement is measured on validation loss, because training loss
                # essentially always keeps falling and would never trigger this.
                if math.isfinite(val_loss) and val_loss < best_loss - 1e-6:
                    best_loss, stale = val_loss, 0
                    history.best_epoch = epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= patience:
                        history.stopped_early = True
                        history.notes.append(
                            f"Stopped at epoch {epoch + 1} of {epochs}: validation loss had "
                            f"not improved for {patience} epochs."
                            + (f" Restored the weights from epoch {history.best_epoch}, where "
                               f"it was {best_loss:.4f}." if best_state else ""))
        if history.stopped_early:
            break

    if patience and best_state is not None and history.best_epoch != len(history.epochs):
        # Finish on the best weights seen, not merely the last ones. Without this,
        # stopping late actively costs accuracy rather than just costing time.
        model.load_state_dict(best_state)
        if not history.stopped_early:
            history.notes.append(
                f"Kept the weights from epoch {history.best_epoch}, the best validation loss; "
                f"the last {len(history.epochs) - history.best_epoch} epoch(s) were worse.")
    elif not patience:
        history.best_epoch = len(history.epochs)

    history.seconds = time.time() - started
    _add_diagnosis(history)
    model.eval()
    return history


def _add_diagnosis(history: History) -> None:
    """Say the useful thing about the curve, so the user does not have to know to look."""
    if len(history.train_loss) < 3:
        return
    train_end, val_end = history.train_loss[-1], history.val_loss[-1]
    val_min = min(history.val_loss)
    if history.train_loss[-1] > history.train_loss[0] * 0.98:
        history.notes.append(
            "The training loss barely moved. Try a higher learning rate, more epochs, or a "
            "bigger model — and check there is an activation between the Linear layers.")
    elif val_end > val_min * 1.15 and train_end < history.train_loss[0] * 0.6:
        history.notes.append(
            "The validation loss turned back up while the training loss kept falling: the "
            "model is memorising the training set. "
            + ("Early stopping already handed you the weights from before that happened, so "
               "this run is fine — but it is the sign to add Dropout, reduce the size, or get "
               "more data." if history.restored else
               "Add Dropout, reduce the size, or get more data."))
    if history.task == "classification" and history.val_acc:
        if history.val_acc[history.kept] > 0.99:
            history.notes.append("Validation accuracy is near perfect — worth checking the task "
                                 "is not easier than intended.")


@torch.no_grad()
def predict(model: CompiledModel, x, device: str = "auto",
            batch_size: int = 512) -> torch.Tensor:
    """Run the model over a tensor — or several, for a multi-input model — without grads."""
    xs = tuple(x) if isinstance(x, (list, tuple)) else (x,)
    dev = resolve_device(device)
    model.to(dev).eval()
    chunks = []
    for start in range(0, xs[0].shape[0], batch_size):
        batch = [t[start:start + batch_size].to(dev) for t in xs]
        if batch[0].shape[0]:
            chunks.append(model(*batch).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)
