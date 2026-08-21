"""Charts, drawn with PIL.

matplotlib would be easier, but it is not something ComfyUI guarantees, and a node pack
that fails to load because of a plotting library is a bad trade. Everything here uses
Pillow, which ComfyUI already requires.

Drawing happens at 2x and is downsampled at the end, which is the cheapest way to get
smooth lines out of PIL.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

BG = (26, 27, 32)
PANEL = (34, 36, 43)
GRID = (52, 55, 64)
AXIS = (110, 115, 128)
TEXT = (208, 212, 222)
MUTED = (140, 146, 160)

#: Distinguishable in order, and each keeps its meaning across every chart in the pack.
SERIES = [
    (94, 168, 255),    # blue      train
    (255, 138, 96),    # orange    validation
    (120, 214, 148),   # green
    (222, 130, 220),   # magenta
    (240, 202, 96),    # yellow
    (110, 226, 224),   # cyan
    (243, 122, 138),   # red
    (168, 156, 255),   # violet
]

SS = 2  # supersampling factor


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class Chart:
    """A plot area with margins, gridlines and labelled axes."""

    def __init__(self, width: int = 720, height: int = 440, title: str = "",
                 xlabel: str = "", ylabel: str = ""):
        self.w, self.h = width * SS, height * SS
        self.out_size = (width, height)
        self.img = Image.new("RGB", (self.w, self.h), BG)
        self.d = ImageDraw.Draw(self.img)
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.left = 72 * SS
        self.right = self.w - 20 * SS
        self.top = (44 if title else 20) * SS
        self.bottom = self.h - (46 if xlabel else 30) * SS
        self.f_small = _font(11 * SS)
        self.f_label = _font(12 * SS)
        self.f_title = _font(15 * SS)
        self.xmin, self.xmax, self.ymin, self.ymax = 0.0, 1.0, 0.0, 1.0
        self._legend: list[tuple[str, tuple[int, int, int]]] = []

    # -- coordinate mapping -------------------------------------------------
    def set_limits(self, xmin, xmax, ymin, ymax):
        if xmax - xmin < 1e-12:
            xmin, xmax = xmin - 0.5, xmax + 0.5
        if ymax - ymin < 1e-12:
            ymin, ymax = ymin - 0.5, ymax + 0.5
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax

    def px(self, x: float) -> float:
        return self.left + (x - self.xmin) / (self.xmax - self.xmin) * (self.right - self.left)

    def py(self, y: float) -> float:
        return self.bottom - (y - self.ymin) / (self.ymax - self.ymin) * (self.bottom - self.top)

    # -- furniture ----------------------------------------------------------
    def frame(self, x_ticks: int = 6, y_ticks: int = 5, x_int: bool = False, y_fmt: str = "{:.3g}"):
        self.d.rectangle([self.left, self.top, self.right, self.bottom], fill=PANEL)
        for value in _ticks(self.ymin, self.ymax, y_ticks):
            y = self.py(value)
            if not self.top <= y <= self.bottom:
                continue
            self.d.line([self.left, y, self.right, y], fill=GRID, width=SS)
            label = y_fmt.format(value)
            self.d.text((self.left - 8 * SS, y), label, font=self.f_small, fill=MUTED,
                        anchor="rm")
        for value in _ticks(self.xmin, self.xmax, x_ticks, integer=x_int):
            x = self.px(value)
            if not self.left <= x <= self.right:
                continue
            self.d.line([x, self.top, x, self.bottom], fill=GRID, width=SS)
            label = f"{int(round(value))}" if x_int else f"{value:.3g}"
            self.d.text((x, self.bottom + 6 * SS), label, font=self.f_small, fill=MUTED,
                        anchor="ma")
        self.d.rectangle([self.left, self.top, self.right, self.bottom], outline=AXIS, width=SS)
        if self.title:
            self.d.text((self.left, 16 * SS), self.title, font=self.f_title, fill=TEXT, anchor="lm")
        if self.xlabel:
            self.d.text(((self.left + self.right) / 2, self.h - 14 * SS), self.xlabel,
                        font=self.f_label, fill=MUTED, anchor="mm")
        if self.ylabel:
            self._vertical_text(self.ylabel)

    def _vertical_text(self, text: str):
        tmp = Image.new("RGB", (self.bottom - self.top, 20 * SS), BG)
        ImageDraw.Draw(tmp).text((tmp.width / 2, tmp.height / 2), text, font=self.f_label,
                                 fill=MUTED, anchor="mm")
        tmp = tmp.rotate(90, expand=True)
        self.img.paste(tmp, (6 * SS, self.top))

    # -- marks --------------------------------------------------------------
    def line(self, xs: Sequence[float], ys: Sequence[float], colour, label: str = "",
             width: int = 2, dashed: bool = False):
        points = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)
                  if math.isfinite(x) and math.isfinite(y)]
        if len(points) >= 2:
            if dashed:
                for i in range(0, len(points) - 1, 2):
                    self.d.line([points[i], points[i + 1]], fill=colour, width=width * SS)
            else:
                self.d.line(points, fill=colour, width=width * SS, joint="curve")
        elif len(points) == 1:
            self.dot(points[0][0], points[0][1], colour, radius=3, raw=True)
        if label:
            self._legend.append((label, colour))

    def dot(self, x: float, y: float, colour, radius: int = 3, raw: bool = False):
        cx, cy = (x, y) if raw else (self.px(x), self.py(y))
        r = radius * SS
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)

    def legend(self):
        """Right-aligned, and wrapped onto as many rows as it takes.

        A single row was laid out from ``right - total``, which goes negative as soon as the
        entries are wider than the chart -- the leading entries were then drawn off the left
        edge of the canvas and simply never appeared. Silent, and worst on exactly the plots
        that carry the most series.
        """
        if not self._legend:
            return
        pad, box = 8 * SS, 10 * SS
        entries = [(name, colour, self.d.textlength(name, font=self.f_small))
                   for name, colour in self._legend]
        limit = self.right - self.left - 2 * pad
        rows: list[tuple[list, float]] = []
        row, used = [], 0.0
        for item in entries:
            width = item[2] + box + 3 * pad
            if row and used + width > limit:
                rows.append((row, used))
                row, used = [], 0.0
            row.append(item)
            used += width
        if row:
            rows.append((row, used))

        y = self.top + pad
        for row, used in rows:
            x = self.right - used - pad
            self.d.rectangle([x - pad, y - pad / 2, self.right - pad / 2, y + box + pad / 2],
                             fill=(24, 25, 30), outline=GRID, width=SS)
            for name, colour, width in row:
                self.d.rectangle([x, y, x + box, y + box], fill=colour)
                self.d.text((x + box + 6 * SS, y + box / 2), name, font=self.f_small,
                            fill=TEXT, anchor="lm")
                x += box + width + 3 * pad
            y += box + 2 * pad

    def note(self, text: str):
        self.d.text((self.left, self.top + 6 * SS), text, font=self.f_small, fill=MUTED, anchor="la")

    def finish(self) -> Image.Image:
        self.legend()
        return self.img.resize(self.out_size, Image.LANCZOS)


def _ticks(lo: float, hi: float, count: int, integer: bool = False) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if raw <= step:
            break
    if integer:
        step = max(1, round(step))
    start = math.ceil(lo / step) * step
    out, value = [], start
    while value <= hi + step * 1e-9 and len(out) < 40:
        out.append(round(value, 10))
        value += step
    return out


# ---------------------------------------------------------------------------
# The actual charts
# ---------------------------------------------------------------------------

def loss_curve(history, width: int = 720, height: int = 440, log_scale: bool = False) -> Image.Image:
    """Training and validation loss per epoch, with the per-step trace behind them."""
    chart = Chart(width, height, "Loss", "epoch", "loss" + (" (log)" if log_scale else ""))
    epochs = history.epochs or [1]

    def prep(values):
        if not log_scale:
            return list(values)
        return [math.log10(max(v, 1e-12)) for v in values]

    series = [v for v in (history.train_loss + history.val_loss) if math.isfinite(v)]
    if not series:
        chart.set_limits(0, 1, 0, 1)
        chart.frame()
        chart.note("no loss recorded")
        return chart.finish()

    values = prep(series)
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.1 or abs(hi) * 0.1 or 0.1
    chart.set_limits(min(epochs) - 0.02 * len(epochs), max(epochs), lo - pad, hi + pad)
    chart.frame(x_int=True)

    if history.step_loss and len(history.step_loss) > len(epochs):
        per_epoch = len(history.step_loss) / len(epochs)
        xs = [(i + 1) / per_epoch for i in range(len(history.step_loss))]
        chart.line(xs, prep(history.step_loss), (60, 78, 104), width=1)
    if history.train_loss:
        chart.line(epochs, prep(history.train_loss), SERIES[0], "train", width=2)
    if history.val_loss and any(math.isfinite(v) for v in history.val_loss):
        chart.line(epochs, prep(history.val_loss), SERIES[1], "validation", width=2)
    return chart.finish()


def accuracy_curve(history, width: int = 720, height: int = 440) -> Image.Image:
    chart = Chart(width, height, "Accuracy", "epoch", "accuracy")
    epochs = history.epochs or [1]
    if not history.val_acc or not any(math.isfinite(v) for v in history.val_acc):
        chart.set_limits(0, 1, 0, 1)
        chart.frame()
        chart.note("accuracy is only recorded for classification")
        return chart.finish()
    chart.set_limits(min(epochs) - 0.02 * len(epochs), max(epochs), 0.0, 1.02)
    chart.frame(x_int=True, y_fmt="{:.0%}")
    chart.line(epochs, history.train_acc, SERIES[0], "train", width=2)
    chart.line(epochs, history.val_acc, SERIES[1], "validation", width=2)
    best = max(history.val_acc)
    chart.line([min(epochs), max(epochs)], [best, best], (90, 96, 110), width=1, dashed=True)
    chart.note(f"best validation accuracy {best * 100:.1f}%")
    return chart.finish()


@torch.no_grad()
def _as_distribution(v: torch.Tensor) -> torch.Tensor:
    """Read a layer's output as one probability per category.

    A gate has already been through a softmax. Running a second one over probabilities
    does not move the argmax, so the regions would be right, but it flattens the peak --
    and the peak is what the fade draws. A hard 0.97 routing decision would be shaded as
    though the model were unsure. So: pass a distribution through untouched.
    """
    if v.shape[-1] == 1:
        return torch.cat([1 - torch.sigmoid(v), torch.sigmoid(v)], dim=-1)
    if v.min() >= 0 and (v.sum(dim=-1) - 1.0).abs().max() < 1e-3:
        return v
    return torch.softmax(v, dim=-1)


def decision_boundary(model, data, resolution: int = 220, width: int = 560,
                      height: int = 560, layer: str = "") -> Image.Image:
    """The whole learned function, drawn.

    Only possible for two-dimensional inputs, which is exactly why the toy datasets exist.

    Name a `layer` and the background becomes that layer's argmax instead of the model's
    answer -- for a mixture-of-experts gate that is a map of which expert owns which part
    of the plane. The dots stay coloured by their true class either way, so the two can be
    read against each other: an expert region holding one colour has specialised, a region
    holding all of them has only carved up space. The region colours are muted so they
    cannot be mistaken for the class colours they sit under.
    """
    if not data.is_2d_points:
        chart = Chart(width, height, "Decision boundary")
        chart.set_limits(0, 1, 0, 1)
        chart.frame()
        chart.note("only available for 2-dimensional classification data")
        return chart.finish()

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    x = torch.cat([data.x_train, data.x_val]).cpu()
    y = torch.cat([data.y_train, data.y_val]).cpu()
    pad = 0.15
    x0, x1 = x[:, 0].min().item(), x[:, 0].max().item()
    y0, y1 = x[:, 1].min().item(), x[:, 1].max().item()
    sx, sy = (x1 - x0) * pad + 1e-6, (y1 - y0) * pad + 1e-6
    x0, x1, y0, y1 = x0 - sx, x1 + sx, y0 - sy, y1 + sy

    gx = torch.linspace(x0, x1, resolution)
    gy = torch.linspace(y1, y0, resolution)          # top row is the highest y
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij"), dim=-1).reshape(-1, 2)
    grid = grid[:, [1, 0]].contiguous()              # back to (x, y) order

    model.eval()
    logits = []
    for start in range(0, grid.shape[0], 8192):
        batch = grid[start:start + 8192].to(device)
        logits.append((model.forward_to(layer, batch) if layer else model(batch)).cpu())
    logits = torch.cat(logits)
    if logits.dim() != 2:
        chart = Chart(width, height, "Decision boundary")
        chart.set_limits(0, 1, 0, 1)
        chart.frame()
        chart.note(f"{layer} gives one {list(logits.shape[1:])} block per point, "
                   "not one value per category")
        return chart.finish()
    probs = _as_distribution(logits)
    winner = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values

    base = torch.tensor(PANEL, dtype=torch.float32)
    field = torch.zeros(resolution * resolution, 3)
    region = []
    for k in range(probs.shape[-1]):
        colour = torch.tensor(SERIES[k % len(SERIES)], dtype=torch.float32)
        if layer:
            # Muted, so a region cannot be read as the class that shares its colour.
            colour = base + (colour - base) * 0.55
        region.append(tuple(int(c) for c in colour.tolist()))
        field[winner == k] = colour
    # Fade towards the panel colour where the model is unsure: uncertainty becomes visible.
    strength = ((confidence - 1.0 / probs.shape[-1]) / (1 - 1.0 / probs.shape[-1])).clamp(0, 1)
    field = base + (field - base) * (0.25 + 0.45 * strength).unsqueeze(-1)

    surface = Image.fromarray(field.reshape(resolution, resolution, 3).byte().numpy(), "RGB")

    title = f"Decision boundary — {layer}" if layer else "Decision boundary"
    chart = Chart(width, height, title, "feature 1", "feature 2")
    chart.set_limits(x0, x1, y0, y1)
    chart.frame()
    surface = surface.resize((chart.right - chart.left, chart.bottom - chart.top), Image.BILINEAR)
    chart.img.paste(surface, (chart.left, chart.top))
    chart.d.rectangle([chart.left, chart.top, chart.right, chart.bottom], outline=AXIS, width=SS)

    for k in range(int(y.max().item()) + 1):
        pts = x[y == k]
        colour = SERIES[k % len(SERIES)]
        for px, py in pts[:1500].tolist():
            chart.d.ellipse([chart.px(px) - 4 * SS, chart.py(py) - 4 * SS,
                             chart.px(px) + 4 * SS, chart.py(py) + 4 * SS],
                            fill=colour, outline=(16, 17, 20), width=SS)
        name = data.classes[k] if k < len(data.classes) else f"class {k}"
        chart._legend.append((name, colour))
    if layer:
        for k, colour in enumerate(region):
            chart._legend.append((f"{layer} · {k}", colour))
    return chart.finish()


@torch.no_grad()
def regression_surface(model, data, resolution: int = 180, width: int = 900,
                       height: int = 320) -> Image.Image:
    """Two numbers in, one out, drawn as a surface: what it should be, what it learned, and
    the difference.

    ``regression_fit`` draws a curve, which needs a single input. The moment there are two
    there is no curve to draw, and the failure this is usually used to show -- a network
    that cannot represent a product -- is invisible in a loss number. Side by side it is
    obvious: the target is a saddle and the plane the model settled on is flat.

    The first two panels share a colour scale so they can be compared by eye. The third does
    not, because the error is usually much smaller than the signal and its own scale is the
    only way to see any structure left in it.
    """
    from .render import colormap

    panel = max(width // 3, 160)
    if data.x_train.dim() != 2 or data.x_train.shape[1] != 2 or data.task != "regression":
        chart = Chart(width, height, "Surface")
        chart.set_limits(0, 1, 0, 1)
        chart.frame()
        chart.note("only available for regression from exactly two inputs")
        return chart.finish()

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    x = torch.cat([data.x_train, data.x_val]).cpu()
    y = torch.cat([data.y_train, data.y_val]).cpu().reshape(-1)
    lo0, hi0 = x[:, 0].min().item(), x[:, 0].max().item()
    lo1, hi1 = x[:, 1].min().item(), x[:, 1].max().item()
    ga = torch.linspace(lo0, hi0, resolution)
    gb = torch.linspace(hi1, lo1, resolution)
    grid = torch.stack(torch.meshgrid(gb, ga, indexing="ij"), dim=-1).reshape(-1, 2)
    grid = grid[:, [1, 0]].contiguous()

    model.eval()
    pred = torch.cat([model(grid[s:s + 8192].to(device)).cpu()
                      for s in range(0, grid.shape[0], 8192)]).reshape(-1)
    # The target is recomputed from the same grid by nearest-neighbour lookup into the data,
    # because the dataset knows its own answer and this module must not guess the formula.
    from .data import _ARITHMETIC
    truth = None
    for name, (fn, _) in _ARITHMETIC.items():
        if name in (data.name or ""):
            truth = fn(grid[:, :1], grid[:, 1:2]).reshape(-1)
            break
    if truth is None:
        truth = torch.full_like(pred, float("nan"))

    lut = colormap("cold-hot")
    shared = max(truth[torch.isfinite(truth)].abs().max().item() if torch.isfinite(truth).any()
                 else 0.0, pred.abs().max().item(), 1e-6)

    def paint(values, scale):
        # colormap() hands back 0..1, not 0..255. Going straight to .byte() floors the
        # whole table to zero and every panel comes out black.
        v = torch.nan_to_num(values / scale, nan=0.0).clamp(-1, 1)
        idx = ((v + 1) * 0.5 * 255).round().long().clamp(0, 255)
        rgb = (lut[idx] * 255).reshape(resolution, resolution, 3).byte().numpy()
        return Image.fromarray(rgb, "RGB")

    error = (pred - truth).abs()
    finite = error[torch.isfinite(error)]
    err_scale = max(finite.max().item() if finite.numel() else 1.0, 1e-6)
    panels = [("what it should be", paint(truth, shared), f"±{shared:.2f}"),
              ("what it learned", paint(pred, shared), f"±{shared:.2f}"),
              ("how wrong it is", paint(error, err_scale), f"0 to {err_scale:.2f}")]

    out = Image.new("RGB", (panel * 3, height), BG)
    for i, (title, surface, scale_text) in enumerate(panels):
        chart = Chart(panel, height, title, "a", "b")
        chart.set_limits(lo0, hi0, lo1, hi1)
        chart.frame()
        box = (chart.right - chart.left, chart.bottom - chart.top)
        chart.img.paste(surface.resize(box, Image.BILINEAR), (chart.left, chart.top))
        chart.d.rectangle([chart.left, chart.top, chart.right, chart.bottom],
                          outline=AXIS, width=SS)
        chart.d.text((chart.right, chart.top - 6 * SS), scale_text, font=chart.f_small,
                     fill=MUTED, anchor="rb")
        out.paste(chart.finish(), (panel * i, 0))
    return out


@torch.no_grad()
def regression_fit(model, data, width: int = 720, height: int = 440) -> Image.Image:
    """The learned curve drawn over the training points. One input, one output only."""
    if data.task != "regression" or data.x_train.dim() != 2 or data.x_train.shape[1] != 1:
        chart = Chart(width, height, "Fit")
        chart.set_limits(0, 1, 0, 1)
        chart.frame()
        chart.note("only available for single-input regression")
        return chart.finish()

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    x = torch.cat([data.x_train, data.x_val]).cpu()
    y = torch.cat([data.y_train, data.y_val]).cpu()
    x0, x1 = x.min().item(), x.max().item()
    span = (x1 - x0) * 0.05 + 1e-6
    grid = torch.linspace(x0 - span, x1 + span, 400).unsqueeze(1)
    model.eval()
    pred = model(grid.to(device)).cpu().reshape(-1)

    lo = min(y.min().item(), pred.min().item())
    hi = max(y.max().item(), pred.max().item())
    pad = (hi - lo) * 0.1 + 1e-6
    chart = Chart(width, height, "Learned function", "input", "target")
    chart.set_limits(x0 - span, x1 + span, lo - pad, hi + pad)
    chart.frame()
    for px, py in zip(x.reshape(-1).tolist(), y.reshape(-1).tolist()):
        chart.dot(px, py, (100, 108, 126), radius=2)
    chart.line(grid.reshape(-1).tolist(), pred.tolist(), SERIES[0], "model", width=3)
    chart._legend.insert(0, ("data", (100, 108, 126)))
    return chart.finish()


@torch.no_grad()
def reconstruction_grid(model, data, count: int = 8, width: int = 720,
                        height: int = 0) -> Image.Image:
    """Originals on top, what the model rebuilt underneath.

    The whole point of an autoencoder is what survives the squeeze, and a loss number does
    not tell you that. Seeing which digits come back blurred and which come back as a
    different digit entirely does.
    """
    use_val = bool(data.n_val)
    inputs = data.val_inputs if use_val else data.train_inputs
    wanted = data.y_val if use_val else data.y_train
    count = max(1, min(int(count), int(inputs[0].shape[0])))
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()
    given = inputs[0][:count]
    target = wanted[:count]
    rebuilt = model(*[t[:count].to(device) for t in inputs]).cpu()

    if given.dim() != 4:
        return text_card(
            f"{data.name}\ninput {tuple(given.shape[1:])}\n"
            f"output {tuple(rebuilt.shape[1:])}\n\n"
            "Side-by-side pictures are only possible for image-shaped data.",
            width=width, title="Reconstruction")

    # When the target is the input (an autoencoder) a third row would be a duplicate. When
    # it is not — a denoiser, a colouriser — the answer has to be shown or there is nothing
    # to judge the output against.
    same_target = (target.shape == given.shape and torch.allclose(target, given, atol=1e-6))
    rows = [("input", given), ("output", rebuilt)]
    if not same_target:
        rows.append(("target", target))

    def prepare(batch):
        t = batch.detach().float().cpu().clamp(0, 1)
        if t.shape[1] in (1, 3):
            t = t.permute(0, 2, 3, 1)
        if t.shape[-1] == 1:
            t = t.repeat(1, 1, 1, 3)
        return t

    rows = [(label, prepare(batch)) for label, batch in rows]
    th, tw = rows[0][1].shape[1], rows[0][1].shape[2]
    scale = max(1, min(96 // max(th, 1), (width - 40) // max(count * tw, 1)))
    cell_w, cell_h, gap = tw * scale, th * scale, 6
    label_w = 46
    canvas_w = label_w + count * (cell_w + gap) + gap
    canvas_h = 34 + len(rows) * (cell_h + gap) + 14
    img = Image.new("RGB", (canvas_w, height or canvas_h), BG)
    draw = ImageDraw.Draw(img)
    draw.text((12, 10), f"{data.name}   {count} examples", font=_font(12), fill=TEXT)

    for row, (label, batch) in enumerate(rows):
        y = 30 + row * (cell_h + gap)
        draw.text((12, y + cell_h / 2), label, font=_font(11), fill=MUTED, anchor="lm")
        for i in range(count):
            cell = Image.fromarray((batch[i] * 255).byte().numpy(), "RGB")
            cell = cell.resize((cell_w, cell_h), Image.NEAREST)
            img.paste(cell, (label_w + i * (cell_w + gap), y))

    # Measured against the target, which is the only comparison that means anything.
    error = (target - rebuilt).abs().mean().item() if target.shape == rebuilt.shape else \
        float("nan")
    caption = (f"mean absolute error vs target {error:.4f}" if error == error
               else "target and output have different shapes")
    draw.text((12, 30 + len(rows) * (cell_h + gap) + 2), caption, font=_font(11), fill=MUTED)
    return img


def dataset_preview(data, width: int = 560, height: int = 560) -> Image.Image:
    """Look at the data before training on it: a scatter, a curve, or a grid of images."""
    x, y = data.x_train, data.y_train

    if data.is_2d_points:
        pts = torch.cat([data.x_train, data.x_val]).cpu()
        labels = torch.cat([data.y_train, data.y_val]).cpu()
        chart = Chart(width, height, data.name, "feature 1", "feature 2")
        pad = 0.08
        x0, x1 = pts[:, 0].min().item(), pts[:, 0].max().item()
        y0, y1 = pts[:, 1].min().item(), pts[:, 1].max().item()
        sx, sy = (x1 - x0) * pad + 1e-6, (y1 - y0) * pad + 1e-6
        chart.set_limits(x0 - sx, x1 + sx, y0 - sy, y1 + sy)
        chart.frame()
        for k in range(int(labels.max().item()) + 1):
            colour = SERIES[k % len(SERIES)]
            for px, py in pts[labels == k][:2000].tolist():
                chart.dot(px, py, colour, radius=3)
            chart._legend.append((data.classes[k] if k < len(data.classes) else f"class {k}",
                                  colour))
        return chart.finish()

    if data.task == "regression" and x.dim() == 2 and x.shape[1] == 1:
        chart = Chart(width, height, data.name, "input", "target")
        xs = torch.cat([data.x_train, data.x_val]).reshape(-1).cpu()
        ys = torch.cat([data.y_train, data.y_val]).reshape(-1).cpu()
        chart.set_limits(xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item())
        chart.frame()
        for px, py in zip(xs.tolist(), ys.tolist()):
            chart.dot(px, py, SERIES[0], radius=2)
        return chart.finish()

    if x.dim() == 4:                                   # a batch of images
        count = min(64, x.shape[0])
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        tiles = x[:count].detach().cpu()
        # Channels-first unless the last axis is the small one. Not just (1, 3): a diffusion
        # input carries the timestep as extra channels, so five is an ordinary picture here.
        if tiles.shape[1] <= tiles.shape[-1]:
            tiles = tiles.permute(0, 2, 3, 1)
        if tiles.shape[-1] == 1:
            tiles = tiles.repeat(1, 1, 1, 3)
        elif tiles.shape[-1] > 3:
            tiles = tiles[..., :3]                     # show the picture, drop the extras
        if tiles.min() < 0:
            # Patches with their mean removed are centred on zero, and clamping would show
            # every darker-than-average pixel as pure black. Zero becomes mid-grey instead,
            # scaled by one factor across the whole sheet so the tiles stay comparable.
            tiles = tiles / (tiles.abs().max().clamp_min(1e-8) * 2) + 0.5
        th, tw = tiles.shape[1], tiles.shape[2]
        pad = 2
        sheet = torch.zeros(rows * (th + pad) + pad, cols * (tw + pad) + pad, 3)
        for i in range(count):
            r, c = divmod(i, cols)
            sheet[pad + r * (th + pad): pad + r * (th + pad) + th,
                  pad + c * (tw + pad): pad + c * (tw + pad) + tw] = tiles[i].clamp(0, 1)
        img = Image.fromarray((sheet * 255).byte().numpy(), "RGB")
        scale = max(1, min(width // max(img.width, 1), (height - 40) // max(img.height, 1)))
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        canvas = Image.new("RGB", (width, height), BG)
        canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2 + 14))
        ImageDraw.Draw(canvas).text((12, 12), f"{data.name}  —  first {count} examples",
                                    font=_font(12), fill=TEXT)
        return canvas

    return text_card(data.describe(), width=width, height=height, title="Dataset")


def confusion_matrix(predicted: torch.Tensor, actual: torch.Tensor,
                     classes: Sequence[str] = (), width: int = 560,
                     height: int = 560) -> Image.Image:
    """Which classes get mistaken for which. The off-diagonal is where the story is."""
    predicted, actual = predicted.reshape(-1).long(), actual.reshape(-1).long()
    k = int(max(predicted.max().item(), actual.max().item())) + 1
    counts = torch.zeros(k, k, dtype=torch.long)
    for p, a in zip(predicted.tolist(), actual.tolist()):
        counts[a, p] += 1
    names = [classes[i] if i < len(classes) else str(i) for i in range(k)]

    chart = Chart(width, height, "Confusion matrix", "predicted", "actual")
    chart.left = max(chart.left, (28 + 6 * max(len(n) for n in names)) * SS)
    chart.set_limits(0, k, 0, k)
    chart.d.rectangle([chart.left, chart.top, chart.right, chart.bottom], fill=PANEL)
    cell_w = (chart.right - chart.left) / k
    cell_h = (chart.bottom - chart.top) / k
    row_totals = counts.sum(dim=1).clamp(min=1)
    font = _font(max(8, int(min(cell_w, cell_h) / SS / 3.2)) * SS)

    for r in range(k):
        for c in range(k):
            frac = (counts[r, c] / row_totals[r]).item()
            hue = SERIES[0] if r == c else SERIES[6]
            base = torch.tensor(PANEL, dtype=torch.float32)
            colour = base + (torch.tensor(hue, dtype=torch.float32) - base) * min(frac * 1.1, 1.0)
            x0 = chart.left + c * cell_w
            y0 = chart.top + r * cell_h
            chart.d.rectangle([x0, y0, x0 + cell_w, y0 + cell_h],
                              fill=tuple(int(v) for v in colour.tolist()),
                              outline=BG, width=max(1, SS // 2))
            if counts[r, c]:
                chart.d.text((x0 + cell_w / 2, y0 + cell_h / 2), str(int(counts[r, c])),
                             font=font, fill=TEXT if frac > 0.35 else MUTED, anchor="mm")
    for i, name in enumerate(names):
        chart.d.text((chart.left - 8 * SS, chart.top + (i + 0.5) * cell_h), name,
                     font=chart.f_small, fill=MUTED, anchor="rm")
        chart.d.text((chart.left + (i + 0.5) * cell_w, chart.bottom + 6 * SS), name,
                     font=chart.f_small, fill=MUTED, anchor="ma")
    chart.d.rectangle([chart.left, chart.top, chart.right, chart.bottom], outline=AXIS, width=SS)
    correct = int(counts.diagonal().sum().item())
    total = int(counts.sum().item()) or 1
    chart.d.text((chart.left, 16 * SS), f"Confusion matrix   —   {correct}/{total} correct "
                                        f"({correct / total * 100:.1f}%)",
                 font=chart.f_title, fill=TEXT, anchor="lm")
    return chart.finish()


def weight_image(tensor: torch.Tensor, width: int = 560, height: int = 560,
                 title: str = "Weights") -> Image.Image:
    """Draw a weight tensor.

    A 4d convolution weight becomes a grid of little filters; anything else is drawn as a
    heatmap of its first two dimensions. Seeing the first convolution layer turn into edge
    detectors is one of the more convincing moments in learning this material.
    """
    t = tensor.detach().float().cpu()
    if t.dim() == 4:                       # [out, in, kh, kw]
        tiles = t.mean(dim=1) if t.shape[1] > 3 else t.permute(0, 2, 3, 1)
        n = tiles.shape[0]
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        kh, kw = t.shape[2], t.shape[3]
        pad = 1
        grid = torch.zeros(rows * (kh + pad) + pad, cols * (kw + pad) + pad,
                           3 if tiles.dim() == 4 else 1)
        for i in range(n):
            r, c = divmod(i, cols)
            tile = tiles[i]
            tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)
            if tile.dim() == 2:
                tile = tile.unsqueeze(-1)
            grid[pad + r * (kh + pad): pad + r * (kh + pad) + kh,
                 pad + c * (kw + pad): pad + c * (kw + pad) + kw] = tile
        array = (grid.repeat(1, 1, 3) if grid.shape[-1] == 1 else grid).clamp(0, 1)
        img = Image.fromarray((array * 255).byte().numpy(), "RGB")
        caption = f"{title}  —  {n} filters of {t.shape[1]}x{kh}x{kw}"
    else:
        flat = t.reshape(t.shape[0], -1) if t.dim() > 2 else t.reshape(1, -1) if t.dim() == 1 else t
        span = max(flat.abs().max().item(), 1e-8)
        norm = (flat / span).clamp(-1, 1)
        rgb = torch.zeros(norm.shape[0], norm.shape[1], 3)
        pos, neg = norm.clamp(min=0), (-norm).clamp(min=0)
        for i, channel in enumerate(SERIES[0]):
            rgb[..., i] += pos * channel / 255
        for i, channel in enumerate(SERIES[1]):
            rgb[..., i] += neg * channel / 255
        img = Image.fromarray((rgb.clamp(0, 1) * 255).byte().numpy(), "RGB")
        caption = f"{title}  —  {tuple(t.shape)}, blue positive / orange negative"

    scale = max(1, min(width // max(img.width, 1), height // max(img.height, 1)))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    canvas = Image.new("RGB", (width, height), BG)
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2 + 10))
    ImageDraw.Draw(canvas).text((12, 12), caption, font=_font(12), fill=TEXT)
    return canvas


def response_histogram(trained: torch.Tensor, untrained: torch.Tensor | None = None,
                       width: int = 720, height: int = 440,
                       title: str = "How often a filter fires") -> Image.Image:
    """The distribution of filter responses, against the same filters before training.

    Both sides are scaled to unit variance first. Without that the picture would be a
    statement about gain — one curve wide and one narrow — and gain is exactly the thing
    these objectives are built not to care about. Divided through, what is left is the
    shape, which is the whole claim: a spike at zero with long tails means the filter is
    silent almost everywhere and emphatic in a few places.

    The vertical axis is log density, because on a linear one the tails — the rare, large,
    informative responses — are invisible, and the tails are the point.
    """
    span, bins = 5.0, 45

    def curve(values: torch.Tensor):
        v = values.detach().float().flatten().cpu()
        v = (v - v.mean()) / v.std().clamp_min(1e-8)
        edges = torch.linspace(-span, span, bins + 1)
        # Deliberately *not* clamping into range first. Clamping would pile every outlying
        # response into the two end bins, which on a sparse distribution is a lot of them,
        # and the plot would end with two tall spikes that are an artefact of the drawing
        # rather than anything about the filter. histc drops them instead.
        counts = torch.histc(v, bins=bins, min=-span, max=span)
        density = counts / counts.sum().clamp_min(1) / ((2 * span) / bins)
        return ((edges[:-1] + edges[1:]) / 2).tolist(), density.tolist()

    xs, ys = curve(trained)
    series = [(xs, ys, SERIES[0], "trained")]
    if untrained is not None:
        series.append((*curve(untrained), SERIES[1], "untrained"))
    measured = list(series)
    series.append((xs, [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) for x in xs],
                   MUTED, "gaussian"))

    # The vertical range comes from the measured curves only. Including the gaussian would
    # let its tail — around 1e-9 at five standard deviations, far below anything a finite
    # sample can show — set the floor, and the difference the plot exists to show would be
    # squeezed into the top tenth of the picture.
    top = max(v for _, values, _, _ in measured for v in values)
    floor = min((v for _, values, _, _ in measured for v in values if v > 0), default=1e-6)
    chart = Chart(width, height, title, "response, in standard deviations",
                  "how often (log)")
    chart.set_limits(-span, span, math.log10(floor) - 0.1, math.log10(top) + 0.15)
    chart.frame(y_fmt="{:.3g}")
    for cx, cy, colour, label in series:
        pairs = [(x, math.log10(y)) for x, y in zip(cx, cy) if y > 0]
        chart.line([p[0] for p in pairs], [p[1] for p in pairs], colour, label,
                   dashed=label == "gaussian")
    return chart.finish()


def text_card(text: str, width: int = 720, height: int = 0, title: str = "") -> Image.Image:
    """Render text as an image, for nodes that want to show a table on the canvas."""
    font = _font(13)
    lines = text.splitlines() or [""]
    line_h = 18
    height = height or max(120, 32 + line_h * len(lines) + (28 if title else 0))
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    y = 14
    if title:
        d.text((16, y), title, font=_font(15), fill=TEXT)
        y += 28
    for line in lines:
        d.text((16, y), line, font=font, fill=TEXT if not line.startswith(" ") else MUTED)
        y += line_h
    return img


def to_comfy_image(img: Image.Image) -> torch.Tensor:
    """PIL -> the [batch, height, width, channels] float tensor ComfyUI passes around."""
    import numpy as np
    array = np.array(img.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).unsqueeze(0)
