"""Looking inside a running network, and asking it what it wants to see.

Everything here outputs a plain ComfyUI IMAGE batch rather than a labelled chart, which is
the whole point: once activations are a batch of images, the rest of ComfyUI can upscale
them, animate them, use them as ControlNet hints, or feed them to a diffusion model,
without any of those nodes knowing what an activation is.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..core import dream as DR
from ..core import render as R
from ..core.errors import NeurodesError
from ..core.runtime import allocating
from ._helpers import image_result, save_inputs
from .types import Activations, Dataset, Model, category

CAT = category("inspect")

_LAYER_HINT = ("Layer name from the Model Summary table, e.g. conv_block_1. "
               "Leave empty for the last layer that has spatial extent. The node lists "
               "what is available when it runs.")


def _source_tensor(model, images, dataset, example: int):
    """Work out what to push through the network, from whichever input is connected.

    Returns a list, because a model can take more than one thing — the picture and the
    timestep that says how noisy it is, say. A dataset knows all of them; a bare image batch
    only knows the first, which is fine for the ordinary case and reported plainly when it
    is not.
    """
    wanted = len(model.input_shapes)
    # Training leaves the model on the GPU, so the input has to go where the weights are.
    if images is not None:
        if wanted > 1:
            raise NeurodesError(
                f"This model takes {wanted} inputs, and an image batch is only one of them.",
                hint="Connect the dataset instead — it carries every input the model needs.",
            )
        with allocating():
            return [R.image_to_model_input(images, model.input_shapes[0]).to(model.device)], \
                   "image input"
    if dataset is not None:
        pool = dataset.val_inputs if dataset.n_val else dataset.train_inputs
        index = max(0, min(int(example), int(pool[0].shape[0]) - 1))
        with allocating():
            return ([t[index: index + 1].clone().to(model.device) for t in pool],
                    f"{dataset.name} example {index}")
    raise NeurodesError(
        "Nothing to run through the model.",
        hint="Connect either an image batch or a dataset.",
    )


class NeuroCaptureActivations(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroCaptureActivations",
            display_name="Capture Activations",
            category=CAT,
            description="Runs one input through the model and keeps every intermediate "
                        "tensor.\n\nWeights show what a layer *is*; activations show what it "
                        "*does to this particular input*. In ordinary PyTorch this needs a "
                        "forward hook on every module — here the forward pass already holds "
                        "each intermediate in order to feed the next one, so it comes free."
                        "\n\nCapture once and render as many layers as you like from the same "
                        "pass.",
            search_aliases=["activations", "feature maps", "hooks", "intermediate",
                            "what it sees", "inspect"],
            inputs=[
                Model.Input("model"),
                io.Image.Input("images", optional=True,
                               tooltip="Any ComfyUI image batch. Converted to the model's "
                                       "channel count and size automatically."),
                Dataset.Input("dataset", optional=True,
                              tooltip="Use an example from a dataset instead of an image."),
                io.Int.Input("example", default=0, min=0, max=1 << 20,
                             tooltip="Which dataset example to use. Ignored when an image "
                                     "is connected."),
            ],
            outputs=[
                Activations.Output(display_name="activations"),
                io.String.Output(display_name="layers"),
            ],
        )

    @classmethod
    def execute(cls, model, images=None, dataset=None, example: int = 0) -> io.NodeOutput:
        inputs, source = _source_tensor(model, images, dataset, example)
        model.eval()
        with torch.no_grad():
            _, captured = model.forward_capturing(*inputs)
        lines = [f"from {source}", ""]
        lines += [f"{name:<24} {R.describe(tensor)}" for name, tensor in captured.items()]
        text = "\n".join(lines)
        return io.NodeOutput({"tensors": captured, "source": source},
                             ", ".join(captured), ui=ui.PreviewText(text))


class NeuroActivationImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroActivationImage",
            display_name="Activation Image",
            category=CAT,
            description="Renders one captured layer as pictures. No axes, no labels, native "
                        "resolution — the tensor and nothing else.\n\n'batch' gives one image "
                        "per channel, which is what makes this useful to the rest of ComfyUI: "
                        "send it to Video Combine for a clip, to an upscaler, to ControlNet, "
                        "or to img2img. 'sheet' gives a single contact sheet, for when you "
                        "are looking rather than piping.",
            search_aliases=["feature map", "activation", "channels", "visualize", "render"],
            inputs=[
                Activations.Input("activations"),
                io.String.Input("layer", default="", tooltip=_LAYER_HINT),
                io.Combo.Input("layout", options=["batch", "sheet"], default="batch",
                               tooltip="One image per channel, or all of them tiled into one."),
                io.Combo.Input("colormap", options=list(R.COLORMAPS), default="viridis"),
                io.Combo.Input("normalization", options=list(R.NORMALIZERS),
                               default="per image",
                               tooltip="'per image' makes every channel legible. 'whole "
                                       "tensor' keeps them comparable, so a channel that "
                                       "barely fired looks like it barely fired."),
                io.Int.Input("upscale", default=4, min=1, max=64,
                             tooltip="Nearest-neighbour, so a 7x7 map stays honest instead of "
                                     "being blurred into looking like more than it is."),
                io.Int.Input("channel", default=-1, min=-1, max=1 << 16,
                             tooltip="-1 for all channels, or pick one."),
                io.Int.Input("columns", default=0, min=0, max=64, advanced=True,
                             tooltip="Sheet layout only. 0 picks a square-ish grid."),
            ] + save_inputs("activations"),
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="stats"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, activations, layer: str = "", layout: str = "batch",
                colormap: str = "viridis", normalization: str = "per image",
                upscale: int = 4, channel: int = -1, columns: int = 0,
                save: bool = False, filename_prefix: str = "") -> io.NodeOutput:
        tensors = activations["tensors"]
        name = _pick_layer(tensors, layer)
        tensor = tensors[name]
        try:
            images = R.to_images(tensor, layout=layout, colormap_name=colormap,
                                 normalization=normalization, upscale=upscale,
                                 channel=int(channel), columns=int(columns))
        except IndexError as exc:
            raise NeurodesError(
                f"{name}: {exc}",
                hint="Set channel to -1 to render all of them.",
            ) from None
        stats = f"{name}   {R.describe(tensor)}\n{images.shape[0]} image(s) at " \
                f"{images.shape[2]}x{images.shape[1]}"
        return image_result(cls, images, stats, save=bool(save),
                            filename_prefix=filename_prefix)


def _pick_layer(tensors: dict, layer: str) -> str:
    wanted = (layer or "").strip()
    if wanted:
        if wanted not in tensors:
            raise NeurodesError(
                f"No captured layer called {wanted!r}.",
                hint="Available: " + ", ".join(tensors),
            )
        return wanted
    spatial = [n for n, t in tensors.items() if t.dim() == 4]
    if spatial:
        return spatial[-1]
    return list(tensors)[-1]


class NeuroDeepDream(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDeepDream",
            display_name="Deep Dream",
            category=CAT,
            description="Asks the network what it wants to see, and draws it.\n\nTraining "
                        "changes the weights to suit the data. This holds the weights still "
                        "and changes the *picture* instead, pushing it uphill until the "
                        "chosen layer is as excited as it can be. So the output is not a "
                        "diagram of what the network learned — it is generated from the "
                        "learned weights themselves.\n\nLeave 'images' unconnected to start "
                        "from noise, which is feature visualisation: the pure form of what "
                        "that layer responds to. Connect an image and it becomes classic "
                        "deep dream, growing the network's obsessions out of your picture.",
            search_aliases=["deepdream", "feature visualization", "inceptionism",
                            "gradient ascent", "what it learned", "psychedelic", "dream"],
            inputs=[
                Model.Input("model"),
                io.String.Input("layer", default="", tooltip=_LAYER_HINT),
                io.Image.Input("images", optional=True,
                               tooltip="Starting picture. Leave empty to start from noise."),
                io.Int.Input("steps", default=20, min=1, max=2000,
                             tooltip="Gradient ascent steps per octave."),
                io.Float.Input("strength", default=0.006, min=0.0005, max=0.2, step=0.001,
                               tooltip="How far to move each step. The gradient is normalised "
                                       "first, so this means much the same thing at any layer. "
                                       "It is really a contrast dial: past about 0.02 the "
                                       "picture saturates to pure black and white."),
                io.Int.Input("channel", default=-1, min=-1, max=1 << 16,
                             tooltip="-1 excites the whole layer. Pick a single channel to "
                                     "see what that one neuron is looking for."),
                io.Int.Input("octaves", default=3, min=1, max=8,
                             tooltip="Optimise small, scale up, optimise again. This is what "
                                     "produces large shapes instead of fine confetti."),
                io.Float.Input("octave_scale", default=1.4, min=1.05, max=2.5, step=0.05,
                               advanced=True),
                io.Float.Input("feature_size", default=0.9, min=0.0, max=1.0, step=0.05,
                               tooltip="How big the shapes come out. Technically it blurs the "
                                       "gradient before applying it: left alone, the optimiser "
                                       "finds that the cheapest way to excite an edge detector "
                                       "is a one-pixel grating, and you get corduroy. Turn "
                                       "this down to 0 to see that happen."),
                io.Int.Input("jitter", default=4, min=0, max=64, advanced=True,
                             tooltip="Random shift before each step, so features cannot lock "
                                     "onto a fixed pixel grid. The other half of the cure "
                                     "for confetti."),
                io.Float.Input("smoothness", default=0.0, min=0.0, max=1.0, step=0.01,
                               advanced=True,
                               tooltip="Penalise neighbouring pixels differing. Cleans up "
                                       "speckle at the cost of detail."),
                io.Combo.Input("objective", options=list(DR.OBJECTIVES), default="mean",
                               advanced=True),
                io.Int.Input("size", default=128, min=32, max=2048,
                             tooltip="Canvas size when starting from noise. Convolutional "
                                     "layers do not care what size they were trained at, so "
                                     "this can be far larger than the model's own input. Note "
                                     "that it does not make the features bigger — their size "
                                     "is set by the layer's receptive field, so a larger "
                                     "canvas simply fits more of them."),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, control_after_generate=True),
            ] + save_inputs("dream"),
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, layer: str = "", images=None, steps: int = 20,
                strength: float = 0.006, channel: int = -1, octaves: int = 3,
                octave_scale: float = 1.4, feature_size: float = 0.9, jitter: int = 4,
                smoothness: float = 0.0, objective: str = "mean", size: int = 128,
                seed: int = 0, save: bool = False,
                filename_prefix: str = "") -> io.NodeOutput:
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        spatial = DR.dreamable_layers(model)
        wanted = (layer or "").strip() or (spatial[-1] if spatial else "")
        if not wanted:
            raise NeurodesError(
                "This model has no layer with spatial extent to dream from.",
                hint="Deep dream needs a convolutional layer. This model appears to be all "
                     "Linear layers, whose activations are vectors with no picture in them.",
            )
        if spatial and wanted not in spatial:
            raise NeurodesError(
                f"'{wanted}' has no spatial extent, so there is no image to grow.",
                hint="Convolutional layers to try: " + ", ".join(spatial),
            )

        shape = model.input_shapes[0]
        with allocating():
            if images is not None:
                canvas = R.image_to_model_input(images, shape, resize=False)
            else:
                canvas = DR.noise_canvas(shape, batch=1, size=int(size), seed=int(seed))

        pbar = ProgressBar(max(1, int(steps) * int(octaves)))

        def on_step(done, total, info):
            pbar.update_absolute(done, total)

        def should_stop():
            mm.throw_exception_if_processing_interrupted()
            return False

        result = DR.dream(
            model, wanted, canvas, steps=int(steps), learning_rate=float(strength),
            channel=int(channel), objective=str(objective), octaves=int(octaves),
            octave_scale=float(octave_scale), jitter=int(jitter),
            feature_scale=float(feature_size), tv_weight=float(smoothness),
            on_step=on_step, should_stop=should_stop)

        out = R.model_input_to_image(result)
        what = "the whole layer" if int(channel) < 0 else f"channel {int(channel)}"
        grey = out[..., 0]
        rails = (((grey < 0.03) | (grey > 0.97)).float().mean() * 100).item()
        report = (f"{model.model_name}: excited {what} of '{wanted}'\n"
                  f"{int(octaves)} octave(s) x {int(steps)} steps, "
                  f"started from {'noise' if images is None else 'the supplied image'}\n"
                  f"output {out.shape[2]}x{out.shape[1]}, {rails:.0f}% of it pure black or white"
                  + ("  (turn 'strength' down for more gradation)" if rails > 60 else "") + "\n"
                  f"spatial layers: {', '.join(spatial)}")
        return image_result(cls, out, report, save=bool(save),
                            filename_prefix=filename_prefix)


VISION_NODES = [NeuroCaptureActivations, NeuroActivationImage, NeuroDeepDream]
