"""Building the model, describing it, and getting it back out as code or weights."""

from __future__ import annotations

import os

from comfy_api.latest import io, ui

from ..core.compile import build_model
from ..core.emit import emit_source
from ..core.errors import GraphError, NeurodesError
from ..core.plot import text_card, to_comfy_image
from ..core.runtime import allocating
from ..core.summary import summarize
from ._helpers import save_image_file, save_inputs
from .types import Model, Tensor, category

CAT = category("model")


class NeuroBuildModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplatePrefix(Tensor.Input("output"), prefix="output",
                                              min=1, max=4)
        return io.Schema(
            node_id="NeuroBuildModel",
            display_name="Build Model",
            category=CAT,
            description="Closes the network. Everything the connected tensors depend on "
                        "becomes one model with real weights, ready to train.\n\nUp to this "
                        "point nothing has been allocated — the graph was only being traced, "
                        "which is why editing it stays instant.",
            search_aliases=["compile", "make model", "finish", "assemble", "end"],
            inputs=[
                io.Autogrow.Input("outputs", template=template,
                                  tooltip="The tensor the network produces. Connect a second "
                                          "one for a model with two heads."),
                io.String.Input("name", default="Model",
                                tooltip="Used for the class name in exported code."),
                io.Boolean.Input("verify", default=True, advanced=True,
                                 tooltip="Push a tiny test batch through after building, to "
                                         "prove the model actually runs. Worth leaving on."),
            ],
            outputs=[
                Model.Output(display_name="model"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, outputs=None, name: str = "Model", verify: bool = True) -> io.NodeOutput:
        tensors = [t for t in (outputs or {}).values() if t is not None]
        if not tensors:
            raise GraphError(
                "Nothing is connected to Build Model.",
                hint="Connect the last tensor of your network to the 'output' slot.",
            )
        model = build_model(tensors, name=name or "Model", verify=bool(verify))
        text = summarize(tensors, name or "Model")
        return io.NodeOutput(model, text, ui=ui.PreviewText(text))


class NeuroModelSummary(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroModelSummary",
            display_name="Model Summary",
            category=CAT,
            description="The layer-by-layer table: what each node produces and how many "
                        "weights it owns. The quickest way to find the layer that is eating "
                        "all your parameters.",
            search_aliases=["summary", "architecture", "parameter count", "table"],
            inputs=[Model.Input("model")] + save_inputs("summary"),
            outputs=[
                io.String.Output(display_name="text"),
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="parameters"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, save: bool = False, filename_prefix: str = "") -> io.NodeOutput:
        text = summarize(model.outputs, model.model_name)
        image = to_comfy_image(text_card(text, width=760))
        if save:
            # The table is the more useful output here, so keep showing it and save quietly.
            save_image_file(cls, image, filename_prefix or "neurodes/summary")
        return io.NodeOutput(text, image, model.n_parameters(), ui=ui.PreviewText(text))


class NeuroExportCode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroExportCode",
            display_name="Export PyTorch Code",
            category=CAT,
            description="Writes the workflow out as an ordinary PyTorch file: a normal "
                        "nn.Module, with every line of forward() annotated with the shape it "
                        "produces.\n\nThis is the point of building networks this way. What "
                        "you drew is not a simulation of a model, it is a model, and here is "
                        "the file to prove it.",
            search_aliases=["export", "python", "source code", "generate code", "save py"],
            inputs=[
                Model.Input("model"),
                io.Boolean.Input("include_demo", default=True,
                                 tooltip="Append a runnable __main__ block that builds the "
                                         "model and prints its parameter count."),
                io.Boolean.Input("save_to_file", default=False,
                                 tooltip="Also write a .py file into ComfyUI's output folder."),
                io.String.Input("filename", default="neurodes_model",
                                tooltip="File name, without the .py."),
            ],
            outputs=[io.String.Output(display_name="python")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, include_demo: bool = True, save_to_file: bool = False,
                filename: str = "neurodes_model") -> io.NodeOutput:
        source = emit_source(model.outputs, name=model.model_name, include_demo=bool(include_demo))
        note = ""
        if save_to_file:
            path = _output_path(filename or "neurodes_model", ".py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            note = f"\n\n# written to {path}"
        return io.NodeOutput(source, ui=ui.PreviewText(source + note))


class NeuroSaveWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroSaveWeights",
            display_name="Save Weights",
            category=CAT,
            description="Writes the trained weights to a .pt file in ComfyUI's output folder. "
                        "Pair it with Export PyTorch Code and the model runs anywhere torch does.",
            search_aliases=["save model", "checkpoint", "state dict", "export weights"],
            inputs=[
                Model.Input("model"),
                io.String.Input("filename", default="neurodes_weights"),
            ],
            outputs=[io.String.Output(display_name="path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, filename: str = "neurodes_weights") -> io.NodeOutput:
        import torch
        path = _output_path(filename or "neurodes_weights", ".pt")
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path)
        note = f"saved {model.n_parameters():,} parameters ({size / 1e6:.2f} MB)\n{path}"
        return io.NodeOutput(path, ui=ui.PreviewText(note))


class NeuroLoadWeights(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroLoadWeights",
            display_name="Load Weights",
            category=CAT,
            description="Loads weights saved earlier back into a model with the same "
                        "architecture. The graph has to match — that is what makes the file "
                        "meaningful.",
            search_aliases=["load model", "restore", "checkpoint"],
            inputs=[
                Model.Input("model"),
                io.String.Input("path", default="",
                                tooltip="Full path to the .pt file, or just its name if it is "
                                        "in ComfyUI's output folder."),
            ],
            outputs=[Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, path: str = "") -> io.NodeOutput:
        import torch
        resolved = path.strip()
        if not resolved:
            raise NeurodesError("No path given.", hint="Type the name of a saved .pt file.")
        if not os.path.isabs(resolved):
            resolved = _output_path(resolved.removesuffix(".pt"), ".pt", unique=False)
        if not os.path.exists(resolved):
            raise NeurodesError(f"No file at {resolved}.",
                                hint="Check the name, or use the path Save Weights returned.")
        with allocating():
            state = torch.load(resolved, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
        note = f"loaded {resolved}"
        if missing or unexpected:
            note += (f"\n{len(missing)} weight(s) in the model were not in the file, "
                     f"{len(unexpected)} in the file were not in the model.\n"
                     "The architecture has probably changed since it was saved.")
        return io.NodeOutput(model, ui=ui.PreviewText(note))


def _output_path(stem: str, suffix: str, unique: bool = True) -> str:
    """A path inside ComfyUI's output folder, falling back to the working directory."""
    try:
        import folder_paths
        directory = folder_paths.get_output_directory()
    except Exception:
        directory = os.getcwd()
    directory = os.path.join(directory, "neurodes")
    os.makedirs(directory, exist_ok=True)
    stem = "".join(c for c in str(stem) if c.isalnum() or c in "-_. ").strip() or "neurodes"
    path = os.path.join(directory, stem + suffix)
    if unique:
        n = 2
        while os.path.exists(path):
            path = os.path.join(directory, f"{stem}_{n}{suffix}")
            n += 1
    return path


MODEL_NODES = [NeuroBuildModel, NeuroModelSummary, NeuroExportCode,
               NeuroSaveWeights, NeuroLoadWeights]
