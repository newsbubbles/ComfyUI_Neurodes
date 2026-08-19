"""Pictures.

Every one of these turns something invisible into something you can look at, because that
is the fastest way to understand what a network is doing. Each draws itself on its own node
and also outputs a ComfyUI IMAGE, and each can write a PNG straight to the output folder,
because in this pack the pictures are often the thing you were after.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..core import discover as DS
from ..core import plot as PL
from ..core import train as T
from ..core.errors import NeurodesError
from ._helpers import image_result, save_inputs
from .types import Dataset, History, Model, category

CAT = category("view")


def _shown(cls, img, *extra, save=False, filename_prefix="") -> io.NodeOutput:
    """Return the chart as an IMAGE *and* draw it on the node itself.

    A plot node that shows nothing until you wire a Preview Image to it is a plot node you
    have to assemble before you can look at anything.
    """
    return image_result(cls, PL.to_comfy_image(img), *extra,
                        save=bool(save), filename_prefix=filename_prefix)


class NeuroPlotLoss(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotLoss",
            display_name="Plot Loss",
            category=CAT,
            description="The training curve. Train and validation loss per epoch, with the "
                        "noisy per-batch trace behind them.\n\nWhat to look for: both falling "
                        "together is healthy; validation turning back up while training keeps "
                        "falling is overfitting; a flat line means nothing is being learned.",
            search_aliases=["loss curve", "training graph", "learning curve", "chart"],
            inputs=[
                History.Input("history"),
                io.Boolean.Input("log_scale", default=False,
                                 tooltip="Plot the logarithm of the loss. Makes late, small "
                                         "improvements visible."),
                io.Int.Input("width", default=720, min=240, max=2048, advanced=True),
                io.Int.Input("height", default=440, min=200, max=2048, advanced=True),
            ] + save_inputs("loss"),
            outputs=[io.Image.Output(display_name="image")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, history, log_scale: bool = False, width: int = 720, height: int = 440,
                save: bool = False, filename_prefix: str = "") -> io.NodeOutput:
        return _shown(cls, PL.loss_curve(history, width, height, bool(log_scale)),
                      save=save, filename_prefix=filename_prefix)


class NeuroPlotAccuracy(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotAccuracy",
            display_name="Plot Accuracy",
            category=CAT,
            description="How often the model is right, per epoch, on both splits. "
                        "Classification only; for regression the loss curve is the whole story.",
            search_aliases=["accuracy graph", "score curve"],
            inputs=[
                History.Input("history"),
                io.Int.Input("width", default=720, min=240, max=2048, advanced=True),
                io.Int.Input("height", default=440, min=200, max=2048, advanced=True),
            ] + save_inputs("accuracy"),
            outputs=[io.Image.Output(display_name="image")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, history, width: int = 720, height: int = 440, save: bool = False,
                filename_prefix: str = "") -> io.NodeOutput:
        return _shown(cls, PL.accuracy_curve(history, width, height),
                      save=save, filename_prefix=filename_prefix)


class NeuroPlotBoundary(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotBoundary",
            display_name="Decision Boundary",
            category=CAT,
            description="Draws the model's entire learned function, by asking it about every "
                        "point on the plane. The colour is the class it would predict, and it "
                        "fades where the model is unsure.\n\nOnly possible when the input is "
                        "two numbers, which is exactly why the Toy Dataset node exists. Add a "
                        "layer, retrain, and watch the boundary bend.",
            search_aliases=["boundary", "what it learned", "visualise model", "regions"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset"),
                io.Int.Input("resolution", default=220, min=40, max=600, advanced=True,
                             tooltip="How finely the plane is sampled."),
                io.Int.Input("size", default=560, min=240, max=1600, advanced=True),
            ] + save_inputs("boundary") + [
                # Appended, not inserted: widget values are stored by position, so anything
                # added ahead of these would shift every value in the saved examples.
                io.String.Input("layer", default="", advanced=True,
                                tooltip="Leave empty for the model's own answer.\n\nName a "
                                        "layer — softmax_1, say — and the background becomes "
                                        "that layer's argmax instead. On a mixture of experts "
                                        "that draws which expert owns which part of the "
                                        "plane. The dots stay coloured by true class, so a "
                                        "region holding one colour has specialised and a "
                                        "region holding all of them has only carved up "
                                        "space."),
            ],
            outputs=[io.Image.Output(display_name="image")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, resolution: int = 220, size: int = 560,
                save: bool = False, filename_prefix: str = "",
                layer: str = "") -> io.NodeOutput:
        layer = (layer or "").strip()
        if layer:
            model._step_named(layer)      # fail here, with the list of names
        return _shown(cls, PL.decision_boundary(model, dataset, int(resolution),
                                                int(size), int(size), layer=layer),
                      save=save, filename_prefix=filename_prefix)


class NeuroPlotFit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotFit",
            display_name="Plot Fit",
            category=CAT,
            description="The curve the model learned, drawn over the data it learned from. "
                        "Single-input regression only. Underfitting and overfitting are both "
                        "obvious here in a way no metric makes them.",
            search_aliases=["regression plot", "fitted curve", "prediction plot"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset"),
                io.Int.Input("width", default=720, min=240, max=2048, advanced=True),
                io.Int.Input("height", default=440, min=200, max=2048, advanced=True),
            ] + save_inputs("fit"),
            outputs=[io.Image.Output(display_name="image")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, width: int = 720, height: int = 440,
                save: bool = False, filename_prefix: str = "") -> io.NodeOutput:
        return _shown(cls, PL.regression_fit(model, dataset, width, height),
                      save=save, filename_prefix=filename_prefix)


class NeuroPlotConfusion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotConfusion",
            display_name="Confusion Matrix",
            category=CAT,
            description="Which classes get mistaken for which. Rows are the truth, columns "
                        "are the guess, so the diagonal is where it was right.\n\nA single "
                        "accuracy number hides everything interesting; this shows you that "
                        "the model confuses 4 with 9 and nothing else.",
            search_aliases=["confusion", "errors", "mistakes", "per class"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset"),
                io.Combo.Input("split", options=["validation", "train"], default="validation"),
                io.Int.Input("size", default=560, min=240, max=1600, advanced=True),
            ] + save_inputs("confusion"),
            outputs=[
                io.Image.Output(display_name="image"),
                io.Float.Output(display_name="accuracy"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, split: str = "validation", size: int = 560,
                save: bool = False, filename_prefix: str = "") -> io.NodeOutput:
        if dataset.task != "classification":
            raise NeurodesError(
                "A confusion matrix only makes sense for classification.",
                hint="This dataset has continuous targets, so use Plot Fit instead.",
            )
        validation = split == "validation"
        x = dataset.val_inputs if validation else dataset.train_inputs
        y = dataset.y_val if validation else dataset.y_train
        logits = T.predict(model, x)
        predicted = (logits.reshape(-1) > 0).long() if logits.shape[-1] == 1 else logits.argmax(-1)
        accuracy = (predicted == y.reshape(-1).cpu()).float().mean().item()
        image = PL.confusion_matrix(predicted, y.cpu(), dataset.classes, int(size), int(size))
        return _shown(cls, image, float(accuracy), save=save, filename_prefix=filename_prefix)


class NeuroPlotWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotWeights",
            display_name="View Weights",
            category=CAT,
            description="Draws a layer's weights. A first convolution layer usually turns "
                        "into edge and blob detectors during training, and seeing that happen "
                        "is more convincing than being told it does.\n\nLeave the name empty "
                        "for the first layer that has weights. The node lists the available "
                        "names when it runs.",
            search_aliases=["filters", "kernels", "weights", "inspect", "features"],
            inputs=[
                Model.Input("model"),
                io.String.Input("layer", default="",
                                tooltip="Layer name from the Model Summary table, e.g. conv2d_1."),
                io.Int.Input("size", default=560, min=240, max=1600, advanced=True),
            ] + save_inputs("weights"),
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="available layers"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, layer: str = "", size: int = 560, save: bool = False,
                filename_prefix: str = "") -> io.NodeOutput:
        ordered, seen = [], set()
        for step in model.plan:
            module = model.module_for(step.op)
            if module is None or step.name in seen:
                continue
            if any(True for _ in module.parameters()):
                seen.add(step.name)
                ordered.append(step.name)
        if not ordered:
            raise NeurodesError(
                "This model has no layers with weights to look at.",
                hint="Add a Linear or Conv 2D layer.",
            )
        wanted = (layer or "").strip() or ordered[0]
        if wanted not in ordered:
            raise NeurodesError(
                f"There is no layer with weights called {wanted!r}.",
                hint="Available: " + ", ".join(ordered),
            )
        module = next(model.module_for(step.op) for step in model.plan if step.name == wanted)
        weight = getattr(module, "weight", None)
        if weight is None:
            weight = next(module.parameters())
        image = PL.weight_image(weight, int(size), int(size), title=f"{wanted}.weight")
        return _shown(cls, image, ", ".join(ordered), save=save,
                      filename_prefix=filename_prefix)


class NeuroPlotDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotDataset",
            display_name="View Dataset",
            category=CAT,
            description="Look at the data before training on it: a scatter for points, the "
                        "curve for regression, a contact sheet for images. Half of all "
                        "training problems are visible here.",
            search_aliases=["preview data", "scatter", "show dataset", "samples"],
            inputs=[
                Dataset.Input("dataset"),
                io.Int.Input("size", default=560, min=240, max=1600, advanced=True),
            ] + save_inputs("dataset"),
            outputs=[io.Image.Output(display_name="image")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, dataset, size: int = 560, save: bool = False,
                filename_prefix: str = "") -> io.NodeOutput:
        return _shown(cls, PL.dataset_preview(dataset, int(size), int(size)),
                      save=save, filename_prefix=filename_prefix)


class NeuroPlotReconstruction(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotReconstruction",
            display_name="Reconstructions",
            category=CAT,
            description="What an autoencoder gives back. Originals on top, the model's "
                        "rebuild underneath.\n\nA loss number cannot tell you *what* was "
                        "lost in the squeeze. This can: whether digits come back blurred, or "
                        "come back as a different digit entirely, is the whole story of what "
                        "the narrow middle decided to keep.",
            search_aliases=["autoencoder", "reconstruction", "before after", "rebuild",
                            "compare"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset"),
                io.Int.Input("count", default=8, min=1, max=32,
                             tooltip="How many examples to show."),
                io.Int.Input("width", default=720, min=240, max=2048, advanced=True),
            ] + save_inputs("reconstruction"),
            outputs=[
                io.Image.Output(display_name="image"),
                io.Float.Output(display_name="mean error"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, count: int = 8, width: int = 720, save: bool = False,
                filename_prefix: str = "") -> io.NodeOutput:
        image = PL.reconstruction_grid(model, dataset, int(count), int(width))
        given = dataset.val_inputs if dataset.n_val else dataset.train_inputs
        wanted = dataset.y_val if dataset.n_val else dataset.y_train
        n = max(1, min(int(count), int(given[0].shape[0])))
        with torch.no_grad():
            rebuilt = model(*[t[:n].to(model.device) for t in given]).cpu()
        # Against the target, not the input: for a denoiser the input is the noisy picture.
        target = wanted[:n].cpu()
        error = ((target - rebuilt).abs().mean().item()
                 if target.shape == rebuilt.shape else float("nan"))
        return _shown(cls, image, float(error), save=save, filename_prefix=filename_prefix)


class NeuroTextCard(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroTextCard",
            display_name="Text Card",
            category=CAT,
            description="Renders text as an image, so a summary or a training report can sit "
                        "on the canvas next to the charts, or be saved with them.",
            search_aliases=["text to image", "caption", "label", "note"],
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input("title", default=""),
                io.Int.Input("width", default=720, min=200, max=2048, advanced=True),
            ] + save_inputs("card"),
            outputs=[io.Image.Output(display_name="image")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, text: str = "", title: str = "", width: int = 720, save: bool = False,
                filename_prefix: str = "") -> io.NodeOutput:
        return _shown(cls, PL.text_card(text or "", width=int(width), title=title),
                      save=save, filename_prefix=filename_prefix)


class NeuroPlotResponses(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPlotResponses",
            display_name="Response Histogram",
            category=CAT,
            description="How often a filter fires, and how hard — drawn against the same "
                        "filters before they were trained.\n\nThis is the picture the "
                        "unsupervised objectives are actually optimising. A trained filter "
                        "gives a tall spike at zero with long thin tails: silent on almost "
                        "every patch, emphatic on a few. An untrained one gives a bell.\n\n"
                        "Both curves are scaled to the same width first, because otherwise "
                        "this would be a picture of gain, and gain is the thing these "
                        "objectives are built to ignore. The vertical axis is logarithmic "
                        "because the rare large responses are the informative ones and a "
                        "linear axis hides them completely.",
            search_aliases=["histogram", "distribution", "sparsity", "kurtosis", "responses",
                            "firing"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset"),
                io.Int.Input("kernel", default=-1, min=-1, max=1 << 14,
                             tooltip="-1 pools every kernel together; or pick one."),
                io.Int.Input("width", default=720, min=320, max=1600, advanced=True),
                io.Int.Input("height", default=440, min=240, max=1200, advanced=True),
            ] + save_inputs("responses"),
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, kernel: int = -1, width: int = 720, height: int = 440,
                save: bool = False, filename_prefix: str = "") -> io.NodeOutput:
        after, before = DS.response_pair(model, dataset, kernel=int(kernel))
        which = "all kernels" if kernel < 0 else f"kernel {kernel}"
        image = PL.response_histogram(after, before, int(width), int(height),
                                      title=f"How often a filter fires — {which}")
        report = DS.report(model, dataset)
        return _shown(cls, image, report, save=save, filename_prefix=filename_prefix)


VIZ_NODES = [NeuroPlotLoss, NeuroPlotAccuracy, NeuroPlotBoundary, NeuroPlotFit,
             NeuroPlotConfusion, NeuroPlotWeights, NeuroPlotDataset,
             NeuroPlotReconstruction, NeuroPlotResponses, NeuroTextCard]
