"""Datasets: something to actually train on."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..core import data as D
from ..core import diffuse as DF
from ..core import discover as DS
from ..core import prepare as P
from ..core import text as TX
from ..core.errors import NeurodesError
from ..core.shape import Shape
from .types import Dataset, ShapeType, category

CAT = category("data")


class NeuroToyDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroToyDataset",
            display_name="Toy Dataset",
            category=CAT,
            description="Two-dimensional points arranged in a named pattern. Nothing to "
                        "download, trains in seconds, and — because the input is only two "
                        "numbers — the network's entire learned function can be drawn as a "
                        "picture by the Decision Boundary node.\n\nStart with 'xor': no "
                        "straight line can separate it, so a network with no hidden layer "
                        "will sit at 50% forever. That failure is worth seeing.",
            search_aliases=["moons", "spirals", "xor", "circles", "blobs", "toy data",
                            "sample data", "demo"],
            inputs=[
                io.Combo.Input("pattern", options=list(D.TOY_NAMES), default="two moons",
                               tooltip="How the points are arranged, and therefore how hard "
                                       "the problem is."),
                io.Int.Input("points", default=1000, min=16, max=200000,
                             tooltip="How many examples to make."),
                io.Float.Input("noise", default=0.1, min=0.0, max=1.0, step=0.01,
                               tooltip="How much the points are scattered. More noise means "
                                       "the classes overlap and perfect accuracy is impossible."),
                io.Float.Input("validation_split", default=0.25, min=0.05, max=0.9, step=0.05,
                               tooltip="Fraction held back to check the model on data it "
                                       "never trained on."),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                io.Int.Output(display_name="classes"),
            ],
        )

    @classmethod
    def execute(cls, pattern: str, points: int, noise: float, validation_split: float,
                seed: int) -> io.NodeOutput:
        bundle = D.toy_classification(pattern, points, noise, validation_split, seed)
        return io.NodeOutput(bundle, bundle.input_shape, bundle.n_classes,
                             ui=ui.PreviewText(bundle.describe()))


class NeuroArithmeticDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroArithmeticDataset",
            display_name="Arithmetic Dataset",
            category=CAT,
            description="Two numbers in, one number out — the smallest problem that still "
                        "needs a hidden layer.\n\nChange the operation and the difficulty "
                        "changes with it. 'a + b' is a weighted sum of the inputs, which is "
                        "exactly what a Linear layer computes, so a network with no hidden "
                        "layer solves it outright. 'a × b' is not, and the same network "
                        "cannot get near it. One combo turns a solved problem into an "
                        "impossible one without touching the network.",
            search_aliases=["multiply", "times", "product", "arithmetic", "regression",
                            "two inputs", "learn to multiply"],
            inputs=[
                io.Combo.Input("operation", options=list(D._ARITHMETIC), default="a × b",
                               tooltip="What the network has to work out from the two "
                                       "numbers it is given."),
                io.Int.Input("points", default=2000, min=64, max=200000),
                io.Float.Input("noise", default=0.0, min=0.0, max=2.0, step=0.01,
                               tooltip="Noise added to the answer, not the inputs. Leave at "
                                       "0 to see how exactly the network can do this."),
                io.Float.Input("span", default=2.0, min=0.5, max=10.0, step=0.5,
                               tooltip="Inputs are drawn from -span to +span."),
                io.Float.Input("validation_split", default=0.25, min=0.05, max=0.5,
                               step=0.05),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
            ],
        )

    @classmethod
    def execute(cls, operation: str, points: int, noise: float, span: float,
                validation_split: float, seed: int) -> io.NodeOutput:
        bundle = D.arithmetic(operation, points, noise, span, validation_split, seed)
        return io.NodeOutput(bundle, bundle.input_shape,
                             ui=ui.PreviewText(bundle.describe()))


class NeuroCurveDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroCurveDataset",
            display_name="Curve Dataset",
            category=CAT,
            description="One number in, one number out: fit a curve. Regression rather than "
                        "classification, so the network predicts a value instead of picking a "
                        "class, and the Fit Curve node draws what it learned over the data.",
            search_aliases=["regression", "sine", "fit curve", "function approximation"],
            inputs=[
                io.Combo.Input("curve", options=list(D.CURVE_NAMES), default="sine",
                               tooltip="The function hiding under the noise."),
                io.Int.Input("points", default=600, min=16, max=200000),
                io.Float.Input("noise", default=0.05, min=0.0, max=1.0, step=0.01),
                io.Float.Input("validation_split", default=0.25, min=0.05, max=0.9, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
            ],
        )

    @classmethod
    def execute(cls, curve: str, points: int, noise: float, validation_split: float,
                seed: int) -> io.NodeOutput:
        bundle = D.toy_regression(curve, points, noise, validation_split, seed)
        return io.NodeOutput(bundle, bundle.input_shape, ui=ui.PreviewText(bundle.describe()))


class NeuroVisionDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroVisionDataset",
            display_name="Image Dataset",
            category=CAT,
            description="A slice of a standard image dataset. Downloads once, on first use, "
                        "and needs torchvision.\n\nThe slice is deliberate: a few thousand "
                        "examples is plenty to watch a network learn, and it keeps a run "
                        "inside a coffee break on a processor with no GPU.",
            search_aliases=["mnist", "cifar", "fashion mnist", "digits", "real data"],
            inputs=[
                io.Combo.Input("dataset", options=list(D.VISION_NAMES), default="MNIST"),
                io.Int.Input("train_examples", default=6000, min=100, max=60000,
                             tooltip="How many training examples to load."),
                io.Int.Input("validation_examples", default=1000, min=100, max=10000),
                io.Boolean.Input("flatten", default=False,
                                 tooltip="On: each image becomes one long row, for a plain "
                                         "Linear network. Off: keep [channels, height, width] "
                                         "for convolutions."),
                io.Boolean.Input("download", default=True, advanced=True,
                                 tooltip="Allow the first-use download."),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                io.Int.Output(display_name="classes"),
            ],
        )

    @classmethod
    def execute(cls, dataset: str, train_examples: int, validation_examples: int,
                flatten: bool, download: bool) -> io.NodeOutput:
        bundle = D.vision_dataset(dataset, limit_train=train_examples,
                                  limit_val=validation_examples, flatten=bool(flatten),
                                  download=bool(download))
        return io.NodeOutput(bundle, bundle.input_shape, bundle.n_classes,
                             ui=ui.PreviewText(bundle.describe()))


class NeuroDatasetFromImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDatasetFromImages",
            display_name="Dataset From Images",
            category=CAT,
            description="Turns a batch of ComfyUI images into a training set, so anything "
                        "the rest of ComfyUI can load or generate can be trained on.\n\n"
                        "ComfyUI keeps images as [batch, height, width, channels]; torch "
                        "convolutions want channels first, and this converts them.",
            search_aliases=["custom dataset", "my images", "train on images", "label images"],
            inputs=[
                io.Image.Input("images", tooltip="A batch of images. One label per image."),
                io.String.Input("labels", default="0, 1",
                                tooltip="One whole number per image, separated by commas. A "
                                        "short list repeats to cover the batch."),
                io.String.Input("class_names", default="",
                                tooltip="Optional display names, separated by commas."),
                io.Float.Input("validation_split", default=0.2, min=0.05, max=0.9, step=0.05),
                io.Boolean.Input("channels_first", default=True, advanced=True,
                                 tooltip="Convert to the [batch, channels, height, width] "
                                         "order torch expects. Turn off only if you are "
                                         "flattening anyway."),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                io.Int.Output(display_name="classes"),
            ],
        )

    @classmethod
    def execute(cls, images, labels: str, class_names: str, validation_split: float,
                channels_first: bool) -> io.NodeOutput:
        n = int(images.shape[0])
        try:
            values = [int(p) for p in str(labels).replace(";", ",").split(",") if p.strip()]
        except ValueError:
            raise NeurodesError(
                f"Could not read the labels {labels!r} as whole numbers.",
                hint="Write one number per image, separated by commas, e.g. '0, 1, 1, 0'.",
            ) from None
        if not values:
            raise NeurodesError("No labels given.",
                                hint="Write one whole number per image, e.g. '0, 1, 1, 0'.")
        if len(values) < n:
            values = [values[i % len(values)] for i in range(n)]
        y = torch.tensor(values[:n], dtype=torch.long)
        names = tuple(p.strip() for p in str(class_names).split(",") if p.strip())
        bundle = D.images_to_dataset(images, y, channels_first=bool(channels_first),
                                     val_fraction=float(validation_split), classes=names,
                                     name=f"{n} images")
        return io.NodeOutput(bundle, bundle.input_shape, bundle.n_classes,
                             ui=ui.PreviewText(bundle.describe()))


class NeuroImageFolderDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroImageFolderDataset",
            display_name="Image Folder Dataset",
            category=CAT,
            description="Trains on your own pictures. Point it at a folder that contains one "
                        "subfolder per class:\n\n"
                        "    my_data/\n"
                        "        cat/   img001.png ...\n"
                        "        dog/   img004.png ...\n\n"
                        "Each subfolder is a class, named after the folder. That is the same "
                        "layout torchvision's ImageFolder uses, so a dataset downloaded from "
                        "anywhere will usually just work. Loaded with Pillow, so no extra "
                        "packages are needed.",
            search_aliases=["my images", "folder", "custom dataset", "imagefolder",
                            "train on my own", "directory", "classifier"],
            inputs=[
                io.String.Input("folder", default="sample_images",
                                tooltip="Path to the folder that CONTAINS the class folders, "
                                        "not one of the class folders itself. A bare name is "
                                        "looked for in ComfyUI's input folder and in this "
                                        "pack's examples folder, which is where the bundled "
                                        "'sample_images' lives."),
                io.Int.Input("size", default=64, min=8, max=512,
                             tooltip="Images are squared off to this many pixels. Smaller "
                                     "trains much faster; 32 to 96 is a sensible range."),
                io.Boolean.Input("greyscale", default=False,
                                 tooltip="Drop colour. Three times less data to learn from, "
                                         "and often no worse if shape is what matters."),
                io.Float.Input("validation_split", default=0.2, min=0.05, max=0.9, step=0.05),
                io.Int.Input("max_per_class", default=0, min=0, max=100000,
                             tooltip="Cap how many images to take from each class. 0 takes "
                                     "all of them.", advanced=True),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, advanced=True,
                             control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                io.Int.Output(display_name="classes"),
                io.String.Output(display_name="class names"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder: str = "", **kwargs):
        """Notice when the folder itself changes.

        ComfyUI caches a node's output while its inputs are unchanged, and the *widget*
        values do not change when you drop twenty more photos into the folder. Without
        this, adding training data would quietly have no effect, which is a horrible thing
        to have to debug. Signing the folder's contents makes the cache do the right thing.
        """
        import os

        root = _resolve_folder(folder)
        if not os.path.isdir(root):
            return root
        signature = []
        for current, _dirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith(D.IMAGE_SUFFIXES):
                    try:
                        signature.append(os.path.getmtime(os.path.join(current, name)))
                    except OSError:
                        signature.append(0.0)
        return f"{len(signature)}:{sum(signature):.0f}"

    @classmethod
    def execute(cls, folder: str, size: int = 64, greyscale: bool = False,
                validation_split: float = 0.2, max_per_class: int = 0,
                seed: int = 0) -> io.NodeOutput:
        folder = _resolve_folder(folder)
        bundle = D.image_folder(folder, size=int(size), greyscale=bool(greyscale),
                                val_fraction=float(validation_split),
                                max_per_class=int(max_per_class), seed=int(seed))
        return io.NodeOutput(bundle, bundle.input_shape, bundle.n_classes,
                             ", ".join(bundle.classes),
                             ui=ui.PreviewText(bundle.describe()))


class NeuroDatasetAutoencoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDatasetAutoencoder",
            display_name="Dataset As Autoencoder",
            category=CAT,
            description="Throws the labels away and makes each example its own target.\n\n"
                        "That one change turns any dataset here into an autoencoder problem: "
                        "the network has to squeeze the input through a narrow middle and "
                        "rebuild it on the other side, so whatever survives the squeeze is "
                        "what it decided mattered. No labels required, which is why this is "
                        "the cheapest interesting thing you can do with a pile of images.\n\n"
                        "The model must return the same shape it takes; Train checks that "
                        "before it starts.",
            search_aliases=["autoencoder", "reconstruction", "unsupervised", "vae",
                            "latent", "compress", "self supervised"],
            inputs=[Dataset.Input("dataset")],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
            ],
        )

    @classmethod
    def execute(cls, dataset) -> io.NodeOutput:
        bundle = D.as_autoencoder(dataset)
        return io.NodeOutput(bundle, bundle.input_shape,
                             ui=ui.PreviewText(bundle.describe()))


class NeuroAugmentDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroAugmentDataset",
            display_name="Augment Dataset",
            category=CAT,
            description="Grows the training set with flipped and jittered copies.\n\nThe "
                        "first honest thing that happens when you train on your own few "
                        "hundred photographs is that the validation loss turns back up. This "
                        "is the cheapest fix: it costs no new data and teaches the network "
                        "that a cat is still a cat three degrees to the left.\n\nOnly the "
                        "training split is augmented. Validation is meant to be a fixed "
                        "yardstick, and a yardstick that changes every run measures nothing.",
            search_aliases=["augmentation", "flip", "rotate", "jitter", "more data",
                            "overfitting", "transform"],
            inputs=[
                Dataset.Input("dataset"),
                io.Int.Input("copies", default=2, min=0, max=16,
                             tooltip="How many jittered copies to add. 2 triples the "
                                     "training set."),
                io.Boolean.Input("flip_horizontal", default=True,
                                 tooltip="Mirror left to right. Safe for almost anything "
                                         "except text and handedness."),
                io.Boolean.Input("flip_vertical", default=False,
                                 tooltip="Mirror top to bottom. Wrong for photographs of "
                                         "the world, fine for textures and microscopy."),
                io.Float.Input("rotate", default=12.0, min=0.0, max=180.0, step=1.0,
                               tooltip="Maximum rotation in degrees, either way."),
                io.Float.Input("zoom", default=0.12, min=0.0, max=0.6, step=0.01,
                               tooltip="Maximum scale change, either way."),
                io.Float.Input("shift", default=0.08, min=0.0, max=0.5, step=0.01,
                               tooltip="Maximum translation, as a fraction of the image."),
                io.Float.Input("brightness", default=0.15, min=0.0, max=0.8, step=0.01,
                               tooltip="Maximum brightness change."),
                io.Float.Input("noise", default=0.0, min=0.0, max=0.5, step=0.01,
                               tooltip="Gaussian noise added to the input.", advanced=True),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, advanced=True,
                             control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                io.Int.Output(display_name="train examples"),
            ],
        )

    @classmethod
    def execute(cls, dataset, copies: int = 2, flip_horizontal: bool = True,
                flip_vertical: bool = False, rotate: float = 12.0, zoom: float = 0.12,
                shift: float = 0.08, brightness: float = 0.15, noise: float = 0.0,
                seed: int = 0) -> io.NodeOutput:
        bundle = P.augment(dataset, copies=int(copies),
                           flip_horizontal=bool(flip_horizontal),
                           flip_vertical=bool(flip_vertical), rotate=float(rotate),
                           zoom=float(zoom), shift=float(shift),
                           brightness=float(brightness), noise=float(noise), seed=int(seed))
        return io.NodeOutput(bundle, bundle.n_train, ui=ui.PreviewText(bundle.describe()))


class NeuroDatasetImageTask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDatasetImageTask",
            display_name="Dataset As Image Task",
            category=CAT,
            description="Turns a folder of pictures into an image-to-image problem by "
                        "spoiling a copy.\n\nThe clean image becomes the target and the "
                        "damaged one becomes the input, so a pile of photographs with no "
                        "labels at all can train a denoiser, a deblurrer or a colouriser. "
                        "The supervision is manufactured from the damage, which is why this "
                        "needs no annotation.\n\nThis is what a U-Net wants to be fed.",
            search_aliases=["denoise", "deblur", "colorize", "colourise", "super resolution",
                            "inpaint", "unet", "image to image", "restoration", "pix2pix"],
            inputs=[
                Dataset.Input("dataset"),
                io.Combo.Input("task", options=list(P.IMAGE_TASKS), default="denoise",
                               tooltip="What damage to undo. 'colourise' also changes the "
                                       "channel count, so the model takes 1 and returns 3."),
                io.Float.Input("strength", default=0.25, min=0.0, max=1.0, step=0.01,
                               tooltip="How badly to spoil it. Higher is a harder problem."),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, advanced=True,
                             control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                ShapeType.Output(display_name="target shape"),
            ],
        )

    @classmethod
    def execute(cls, dataset, task: str = "denoise", strength: float = 0.25,
                seed: int = 0) -> io.NodeOutput:
        bundle = P.as_image_task(dataset, task=str(task), strength=float(strength),
                                 seed=int(seed))
        return io.NodeOutput(bundle, bundle.input_shape, bundle.target_shape,
                             ui=ui.PreviewText(bundle.describe()))


class NeuroImagePairsDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroImagePairsDataset",
            display_name="Image Pairs From Folders",
            category=CAT,
            description="Two parallel folders, matched by filename. For data that is "
                        "genuinely paired: photographs and their segmentation masks, before "
                        "and after, sketch and render.\n\nFiles pair on the name without the "
                        "extension, so input/042.jpg goes with target/042.png.",
            search_aliases=["paired", "pix2pix", "segmentation", "before after",
                            "translation", "masks"],
            inputs=[
                io.String.Input("input_folder", default="",
                                tooltip="Folder of images the model will be given."),
                io.String.Input("target_folder", default="",
                                tooltip="Folder of images the model should produce."),
                io.Int.Input("size", default=64, min=8, max=512),
                io.Boolean.Input("greyscale", default=False,
                                 tooltip="Drop colour from the input."),
                io.Boolean.Input("target_greyscale", default=False,
                                 tooltip="Drop colour from the target. Turn this on for "
                                         "single-channel masks."),
                io.Float.Input("validation_split", default=0.2, min=0.05, max=0.9, step=0.05),
                io.Int.Input("limit", default=0, min=0, max=100000,
                             tooltip="Cap how many pairs to load. 0 takes all.",
                             advanced=True),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, advanced=True,
                             control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                ShapeType.Output(display_name="target shape"),
            ],
        )

    @classmethod
    def execute(cls, input_folder: str, target_folder: str, size: int = 64,
                greyscale: bool = False, target_greyscale: bool = False,
                validation_split: float = 0.2, limit: int = 0,
                seed: int = 0) -> io.NodeOutput:
        bundle = P.pairs_from_folders(_resolve_folder(input_folder),
                                      _resolve_folder(target_folder), size=int(size),
                                      greyscale=bool(greyscale),
                                      target_greyscale=bool(target_greyscale),
                                      val_fraction=float(validation_split),
                                      limit=int(limit), seed=int(seed))
        return io.NodeOutput(bundle, bundle.input_shape, bundle.target_shape,
                             ui=ui.PreviewText(bundle.describe()))


def _search_roots() -> list[str]:
    """Where a bare filename is looked for: ComfyUI's input folder, then the pack's own."""
    import os

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    roots = [os.path.join(here, "examples"), here]
    try:
        import folder_paths
        roots.insert(0, folder_paths.get_input_directory())
    except Exception:
        pass
    return roots


def _resolve_folder(folder: str) -> str:
    """Let a relative path mean something sensible.

    An absolute path is used as-is. A bare name like ``sample_images`` is looked for in
    ComfyUI's input folder and in the pack's own examples folder, so the bundled example
    workflow runs on any machine without anyone editing a path first.
    """
    import os

    text = str(folder).strip().strip('"')
    if not text or os.path.isabs(text):
        return text
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [os.path.join(here, "examples", text), os.path.join(here, text)]
    try:
        import folder_paths
        candidates.insert(0, os.path.join(folder_paths.get_input_directory(), text))
    except Exception:
        pass
    candidates.append(os.path.abspath(text))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return text


class NeuroDatasetInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDatasetInfo",
            display_name="Dataset Info",
            category=CAT,
            description="What is in a dataset, and the shape its examples have — connect the "
                        "shape output straight into an Input node and the two can never "
                        "disagree.",
            search_aliases=["dataset shape", "classes", "inspect data"],
            inputs=[Dataset.Input("dataset")],
            outputs=[
                ShapeType.Output(display_name="input shape"),
                ShapeType.Output(display_name="target shape"),
                io.Int.Output(display_name="classes"),
                io.Int.Output(display_name="train examples"),
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(cls, dataset) -> io.NodeOutput:
        return io.NodeOutput(dataset.input_shape, dataset.target_shape, dataset.n_classes,
                             dataset.n_train, dataset.describe(),
                             ui=ui.PreviewText(dataset.describe()))


class NeuroDatasetDiffusion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDatasetDiffusion",
            display_name="Dataset As Diffusion",
            category=CAT,
            description="Turns any folder of pictures into the training problem behind a "
                        "diffusion model.\n\nEach picture is mixed with noise by a random "
                        "amount, and the network's job is to say which part of the mess was "
                        "the noise. That is the whole of diffusion training — an ordinary "
                        "supervised problem that the ordinary Train node can solve.\n\n"
                        "The network is also told how noisy its input is, as extra channels "
                        "carrying the timestep, so the input has more channels than the "
                        "picture does. Wire the 'input shape' output into your Input node "
                        "and it works itself out.",
            search_aliases=["diffusion", "ddpm", "ddim", "denoising diffusion", "generative",
                            "noise prediction", "sampler", "latent diffusion", "stable"],
            inputs=[
                Dataset.Input("dataset"),
                io.Int.Input("copies", default=6, min=1, max=64,
                             tooltip="How many noise levels to draw per picture. The network "
                                     "has to be good at every point of the journey, not only "
                                     "the middle, so each image is used several times."),
                io.Combo.Input("predict", options=list(DF.PREDICTS), default="noise",
                               tooltip="What the network has to produce. 'noise' is what real "
                                       "diffusion models do and is the more surprising fact. "
                                       "'image' asks it to guess the clean picture instead, "
                                       "which is easier to picture and often steadier on a "
                                       "small model. Both sample."),
                io.Combo.Input("timestep", options=list(DF.TIMESTEPS),
                               default="extra channels",
                               tooltip="How the network is told how noisy its input is.\n\n"
                                       "'extra channels' staples the encoding onto the "
                                       "picture, so the model needs nothing but a wider "
                                       "Input.\n\n'second input' hands it over separately, "
                                       "the way a real implementation does — then the graph "
                                       "has to say what to do with it, which means a second "
                                       "Input node and an Add somewhere."),
                io.Combo.Input("schedule", options=list(DF.SCHEDULES), default="cosine",
                               tooltip="How fast the picture is destroyed as t goes 0 to 1. "
                                       "'cosine' spends more of its length near the clean "
                                       "end, where the differences that matter are."),
                io.Int.Input("time_channels", default=2, min=0, max=8, advanced=True,
                             tooltip="How many numbers encode the timestep. 0 hides it — "
                                     "worth trying once, to see a model that cannot tell how "
                                     "far along it is produce mud.", ),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, advanced=True,
                             control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                ShapeType.Output(display_name="target shape"),
                ShapeType.Output(display_name="timestep shape",
                                 tooltip="Wire this into a second Input node when 'timestep' "
                                         "is set to 'second input'."),
            ],
        )

    @classmethod
    def execute(cls, dataset, copies: int = 6, predict: str = "noise",
                timestep: str = "extra channels", schedule: str = "cosine",
                time_channels: int = 2, seed: int = 0) -> io.NodeOutput:
        bundle = DF.as_diffusion_task(dataset, copies=int(copies), predict=str(predict),
                                      schedule=str(schedule), timestep=str(timestep),
                                      time_channels=int(time_channels), seed=int(seed))
        shapes = bundle.input_shapes
        return io.NodeOutput(bundle, shapes[0], bundle.target_shape,
                             shapes[1] if len(shapes) > 1 else Shape(["B", int(time_channels)]),
                             ui=ui.PreviewText(bundle.describe()))


class NeuroTextDataset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroTextDataset",
            display_name="Text Dataset",
            category=CAT,
            description="Text, cut into overlapping windows, each paired with itself shifted "
                        "one character to the left.\n\nThat shift is the entire training "
                        "objective of a language model: at every position, guess the next "
                        "character. Nothing is labelled — the text is its own answer.\n\n"
                        "Characters rather than word tokens, on purpose: the vocabulary is "
                        "small enough to train on a CPU, there is no tokenizer to install, "
                        "and you can watch it learn spaces, then word shapes, then real "
                        "words, in that order.",
            search_aliases=["gpt", "language model", "nanogpt", "char rnn", "text", "corpus",
                            "next token", "llm", "shakespeare", "tokens"],
            inputs=[
                io.String.Input("text", default="", multiline=True,
                                tooltip="Paste text here, or leave it empty and give a file "
                                        "instead. Empty on both uses a short built-in "
                                        "sample."),
                io.String.Input("file", default="sample_text.txt",
                                tooltip="A .txt file. A bare name is looked for in ComfyUI's "
                                        "input folder and in this pack's examples folder, "
                                        "where 'sample_text.txt' lives."),
                io.Int.Input("context", default=64, min=8, max=1024,
                             tooltip="How many characters the model can see at once. This is "
                                     "the sequence length, and attention costs its square."),
                io.Int.Input("stride", default=0, min=0, max=1024,
                             tooltip="How far apart the windows start. 1 uses every position "
                                     "and squeezes the most out of a small text; 'context' "
                                     "uses each character once and is much faster. 0 picks "
                                     "a quarter of the context.", advanced=True),
                io.Float.Input("validation_split", default=0.1, min=0.02, max=0.5, step=0.01,
                               tooltip="Held out from the END of the text, not at random — "
                                       "the windows overlap, so a random split would put "
                                       "nearly the same sequence on both sides."),
                io.Int.Input("limit", default=0, min=0, max=10_000_000,
                             tooltip="Use only the first N characters. 0 takes all of it.",
                             advanced=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                io.Int.Output(display_name="vocabulary"),
                io.String.Output(display_name="characters"),
            ],
        )

    @classmethod
    def execute(cls, text: str = "", file: str = "", context: int = 64, stride: int = 0,
                validation_split: float = 0.1, limit: int = 0) -> io.NodeOutput:
        bundle = TX.text_dataset(text=str(text), path=str(file).strip().strip('"'),
                                 context=int(context), stride=int(stride),
                                 val_fraction=float(validation_split), limit=int(limit),
                                 search=_search_roots())
        return io.NodeOutput(bundle, bundle.input_shape, bundle.n_classes,
                             "".join(bundle.classes),
                             ui=ui.PreviewText(bundle.describe()))


class NeuroDatasetPairs(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDatasetPairs",
            display_name="Dataset As Pairs",
            category=CAT,
            description="Turns a labelled dataset into \"are these two the same thing?\".\n\n"
                        "Two examples in, one bit out. That is the problem a Siamese network "
                        "exists for: it never learns the classes, it learns a comparison, so "
                        "it can be asked about classes it has never seen.\n\nThe two "
                        "examples arrive as two separate inputs, so the model needs two "
                        "Input nodes — and the same `share` tag on both towers, or they see "
                        "the world differently and the comparison means nothing.",
            search_aliases=["siamese", "pairs", "same or different", "contrastive",
                            "verification", "one shot", "metric learning", "similarity"],
            inputs=[
                Dataset.Input("dataset"),
                io.Int.Input("pairs", default=4, min=1, max=64,
                             tooltip="How many pairs to draw per example. Half share a "
                                     "class and half do not, so chance is 50%."),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, advanced=True,
                             control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape",
                                 tooltip="The same shape for both towers — wire it into "
                                         "both Input nodes."),
                io.Int.Output(display_name="train examples"),
            ],
        )

    @classmethod
    def execute(cls, dataset, pairs: int = 4, seed: int = 0) -> io.NodeOutput:
        bundle = P.as_pairs(dataset, pairs=int(pairs), seed=int(seed))
        return io.NodeOutput(bundle, bundle.input_shape, bundle.n_train,
                             ui=ui.PreviewText(bundle.describe()))


class NeuroImagePatches(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroImagePatches",
            display_name="Image Patches",
            category=CAT,
            description="Thousands of little squares cut at random out of one picture, with "
                        "no labels and nothing to predict.\n\nThis is the raw material for "
                        "learning filters rather than being given them: feed it to Discover "
                        "Kernels and a bank of convolutions will go looking for whatever "
                        "structure this particular image has. Change the picture and you get "
                        "a different set of filters, which is the interesting part.\n\n"
                        "Wire 'input shape' straight into an Input node.",
            search_aliases=["patches", "unsupervised", "sparse coding", "ica", "filters",
                            "kernels", "self-supervised", "no labels", "crops"],
            inputs=[
                io.Image.Input("image", tooltip="Any image. One is enough — a photograph has "
                                                "hundreds of thousands of patches in it."),
                io.Int.Input("patch_size", default=12, min=3, max=64,
                             tooltip="The size of each square, and so the largest kernel that "
                                     "can be learned. 12 is big enough to hold an edge and "
                                     "small enough to train in seconds."),
                io.Int.Input("count", default=6000, min=64, max=200000,
                             tooltip="How many patches to cut. More is a better estimate of "
                                     "what this image looks like, at linear cost."),
                io.Boolean.Input("greyscale", default=True,
                                 tooltip="Collapse colour to brightness. On, the kernels are "
                                         "about shape; off, they are free to be about colour "
                                         "too, and some of them will be."),
                io.Boolean.Input("remove_mean", default=True,
                                 tooltip="Subtract each patch's own average brightness.\n\n"
                                         "Not cosmetic. A patch's mean is the biggest thing "
                                         "in it, and a filter that just measures brightness "
                                         "scores well without having learned any structure. "
                                         "Turning this off drops kurtosis from 19.5 to 8.4 "
                                         "and the kernels come out visibly worse."),
                io.Float.Input("val_fraction", default=0.15, min=0.0, max=0.5, step=0.05,
                               advanced=True,
                               tooltip="A strip down the right-hand edge, held out. A strip "
                                       "rather than a random split because patches overlap, "
                                       "and a random split would put nearly the same pixels "
                                       "on both sides of it."),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, advanced=True,
                             control_after_generate=True),
            ],
            outputs=[
                Dataset.Output(display_name="dataset"),
                ShapeType.Output(display_name="input shape"),
                io.Int.Output(display_name="patches"),
            ],
        )

    @classmethod
    def execute(cls, image, patch_size: int = 12, count: int = 6000, greyscale: bool = True,
                remove_mean: bool = True, val_fraction: float = 0.15,
                seed: int = 0) -> io.NodeOutput:
        bundle = DS.image_patches(image, patch=int(patch_size), count=int(count),
                                  greyscale=bool(greyscale), remove_mean=bool(remove_mean),
                                  val_fraction=float(val_fraction), seed=int(seed))
        return io.NodeOutput(bundle, bundle.input_shape, bundle.n_train,
                             ui=ui.PreviewText(bundle.describe()))


DATA_NODES = [NeuroToyDataset, NeuroArithmeticDataset, NeuroCurveDataset, NeuroVisionDataset,
              NeuroDatasetFromImages, NeuroImageFolderDataset, NeuroImagePairsDataset,
              NeuroDatasetAutoencoder, NeuroDatasetImageTask, NeuroDatasetDiffusion,
              NeuroDatasetPairs, NeuroTextDataset, NeuroImagePatches,
              NeuroAugmentDataset, NeuroDatasetInfo]
