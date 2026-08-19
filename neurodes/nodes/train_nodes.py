"""Training, evaluating, and running a trained model on real input."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ..core import diffuse as DF
from ..core import discover as DS
from ..core import render as R
from ..core import text as TX
from ..core import train as T
from ..core.errors import NeurodesError
from ..core.plot import loss_curve
from ..core.runtime import adopt, allocating
from ._helpers import image_result, save_inputs
from .types import Dataset, History, Model, Trainer, category

CAT = category("train")


class NeuroTrainer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroTrainer",
            display_name="Trainer",
            category=CAT,
            description="How to train, kept separate from what to train so one recipe can "
                        "drive several models — which is how you compare architectures fairly."
                        "\n\nIf you change one number, make it the learning rate. It matters "
                        "more than everything else here put together.",
            search_aliases=["optimizer", "hyperparameters", "learning rate", "recipe",
                            "adam", "sgd", "loss"],
            inputs=[
                io.Int.Input("epochs", default=60, min=1, max=100000,
                             tooltip="How many times to go through the whole training set. "
                                     "With early stopping on this is a ceiling rather than a "
                                     "prescription, so it is safe to set it generously."),
                io.Int.Input("early_stopping", default=20, min=0, max=10000,
                             tooltip="Stop after this many epochs with no improvement in "
                                     "validation loss, and keep the best weights rather than "
                                     "the last ones. 0 turns it off.\n\nLoss usually plateaus "
                                     "long before the epoch count runs out, and without this "
                                     "the extra epochs are not merely wasted — they overfit, "
                                     "so the model you end up with is worse than the one you "
                                     "passed through."),
                io.Int.Input("batch_size", default=64, min=1, max=16384,
                             tooltip="How many examples the network sees before each weight "
                                     "update. Bigger is steadier and faster per example; "
                                     "smaller is noisier, which sometimes helps."),
                io.Float.Input("learning_rate", default=0.001, min=1e-8, max=10.0, step=0.0001,
                               tooltip="How big a step to take each update. Too high and the "
                                       "loss explodes; too low and nothing happens. 0.001 is "
                                       "a sane place to start with Adam."),
                io.Combo.Input("optimizer", options=list(T.OPTIMIZERS), default="adam",
                               tooltip="How the steps are chosen. Adam adapts per weight and "
                                       "is forgiving; plain SGD needs a well-chosen rate."),
                io.Combo.Input("loss", options=list(T.LOSSES), default="auto",
                               tooltip="What counts as wrong. 'auto' picks cross entropy for "
                                       "classification and mean squared error for regression, "
                                       "which is right almost always."),
                io.Float.Input("weight_decay", default=0.0, min=0.0, max=1.0, step=0.0001,
                               tooltip="Pulls weights toward zero. A mild cure for "
                                       "overfitting.", advanced=True),
                io.Float.Input("grad_clip", default=0.0, min=0.0, max=100.0, step=0.1,
                               tooltip="Caps the size of an update. 1.0 is a common value for "
                                       "recurrent networks, which otherwise blow up. 0 is off.",
                               advanced=True),
                io.Combo.Input("device", options=["auto", "cpu", "cuda"], default="auto",
                               advanced=True),
                io.Boolean.Input("shuffle", default=True, advanced=True,
                                 tooltip="Reorder the training set each epoch."),
            ],
            outputs=[Trainer.Output(display_name="trainer")],
        )

    @classmethod
    def execute(cls, epochs, early_stopping, batch_size, learning_rate, optimizer, loss,
                weight_decay, grad_clip, device, shuffle) -> io.NodeOutput:
        cfg = T.TrainConfig(epochs=int(epochs), batch_size=int(batch_size),
                            learning_rate=float(learning_rate), optimizer=str(optimizer),
                            loss=str(loss), weight_decay=float(weight_decay),
                            grad_clip=float(grad_clip), device=str(device),
                            shuffle=bool(shuffle), early_stopping=int(early_stopping))
        note = (f"{cfg.optimizer}, lr {cfg.learning_rate}, up to {cfg.epochs} epochs, "
                f"batch {cfg.batch_size}, loss {cfg.loss}\n"
                + (f"stops after {cfg.early_stopping} epochs without improvement"
                   if cfg.early_stopping else "runs all epochs, no early stopping"))
        return io.NodeOutput(cfg, ui=ui.PreviewText(note))


class NeuroTrain(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroTrain",
            display_name="Train",
            category=CAT,
            description="Fits the model to the data.\n\nBefore the first step it checks the "
                        "things that usually go wrong — the wrong number of outputs for the "
                        "number of classes, a softmax in front of a loss that applies its "
                        "own, a regression loss on whole-number labels — and explains them "
                        "instead of training a broken model for ten minutes.\n\nChange the "
                        "seed to train again from scratch.",
            search_aliases=["fit", "learn", "run training", "go", "optimize"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset"),
                Trainer.Input("trainer"),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, control_after_generate=True,
                             tooltip="Makes a run repeatable. Change it to train again with "
                                     "different starting weights."),
                io.Boolean.Input("live_preview", default=True, advanced=True,
                                 tooltip="Draw the loss curve onto the node's progress bar as "
                                         "it trains."),
                io.Boolean.Input("reset_weights", default=True, advanced=True,
                                 tooltip="Draw new weights before training, so the seed "
                                         "really does decide where the run starts.\n\n"
                                         "ComfyUI does not re-run a node whose inputs have "
                                         "not changed, so the model arriving here is the "
                                         "same object the last run trained. With this off, "
                                         "pressing Run twice does not compare two settings "
                                         "— it compares one setting against itself plus "
                                         "more training. Turn it off deliberately to carry "
                                         "on training a model that is already trained."),
            ],
            outputs=[
                Model.Output(display_name="trained model"),
                History.Output(display_name="history"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, trainer, seed: int = 0, live_preview: bool = True,
                reset_weights: bool = True) -> io.NodeOutput:
        import dataclasses

        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        cfg = dataclasses.replace(trainer, seed=int(seed),
                                  reset_weights=bool(reset_weights))
        total_epochs = max(1, int(cfg.epochs))
        pbar = ProgressBar(total_epochs)

        # Handed to train() so the preview can read the curve while it is still being drawn.
        history = T.History()
        state = {"epoch": 0}

        def on_progress(step: int, total: int, info: dict) -> None:
            epoch = int(info.get("epoch", 0))
            if epoch != state["epoch"]:
                state["epoch"] = epoch
                pbar.update_absolute(epoch - 1, total_epochs,
                                     _preview(history) if live_preview else None)

        def should_stop() -> bool:
            mm.throw_exception_if_processing_interrupted()
            return False

        result = T.train(model, dataset, cfg, on_progress=on_progress,
                         should_stop=should_stop, history=history)
        pbar.update_absolute(total_epochs, total_epochs)

        report = f"{model.model_name}: {model.n_parameters():,} parameters\n{result.summary()}"
        return io.NodeOutput(model, result, report, ui=ui.PreviewText(report))


def _preview(history):
    try:
        image = loss_curve(history, width=384, height=256)
        return ("JPEG", image, None)
    except Exception:
        return None


class NeuroDiscover(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroDiscover",
            display_name="Discover Kernels",
            category=CAT,
            description="Trains a bank of convolutions with nothing to copy and no labels — "
                        "only a statement about what a good filter *does*.\n\nFeed it Image "
                        "Patches and a model that ends on a Conv 2D. What comes out, from a "
                        "photograph, is oriented edge detectors at a range of angles and "
                        "sizes: the same alphabet that falls out of sparse coding, of ICA on "
                        "natural scenes, and of the first layer of any convolutional net "
                        "trained on pictures. Nobody said the word 'edge' anywhere.\n\n"
                        "Two settings decide whether that happens at all, and both are worth "
                        "breaking on purpose. Set 'objective' to 'histogram change' — the "
                        "obvious idea — and it learns tiles of static. Set 'diversity' to 0 "
                        "and all the kernels collapse onto the same one while the loss curve "
                        "says everything is fine.",
            search_aliases=["unsupervised", "sparse coding", "ica", "learn filters",
                            "kernels", "self-supervised", "no labels", "gabor", "edge "
                            "detectors", "discover"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset", tooltip="From Image Patches."),
                io.Combo.Input("objective", options=list(DS.OBJECTIVES),
                               default="sparse response",
                               tooltip="What counts as a good filter.\n\n'sparse response': "
                                       "one that is quiet almost everywhere and loud in a "
                                       "few places. This is the one that works.\n\n'peaky "
                                       "response': the same idea measured by kurtosis — "
                                       "sharper, and more easily swayed by a few odd "
                                       "patches.\n\n'histogram change': make the output "
                                       "histogram as unlike the input's as possible. The "
                                       "obvious idea, and a failure worth running once."),
                io.Float.Input("diversity", default=1.0, min=0.0, max=20.0, step=0.1,
                               tooltip="How hard to push the kernels apart, by penalising "
                                       "any two of them for responding to the same patches."
                                       "\n\nWithout this they all find the same filter — "
                                       "and score *better* for it, because sixteen copies of "
                                       "the best answer is sixteen good answers as far as "
                                       "the objective can tell. 1.0 is a good default; above "
                                       "about 3 the kernels start to smear."),
                io.Int.Input("epochs", default=120, min=1, max=100000,
                             tooltip="Passes over the patches. These are tiny models, so "
                                     "this is seconds, and more visibly sharpens the "
                                     "filters."),
                io.Int.Input("batch_size", default=512, min=16, max=16384,
                             tooltip="Both objectives are statistics of a batch — a fourth "
                                     "moment, and a correlation between every pair of "
                                     "kernels. Small batches estimate those badly, so this "
                                     "wants to be much larger than usual."),
                io.Float.Input("learning_rate", default=0.02, min=1e-8, max=10.0, step=0.001),
                io.Int.Input("seed", default=0, min=0, max=1 << 31, control_after_generate=True),
                io.Int.Input("early_stopping", default=0, min=0, max=10000, advanced=True,
                             tooltip="Off by default: there is no target to overfit to here, "
                                     "and the filters keep sharpening long after the loss "
                                     "has mostly flattened."),
                io.Boolean.Input("live_preview", default=True, advanced=True),
                io.Boolean.Input("reset_weights", default=True, advanced=True,
                                 tooltip="Draw new kernels before training. Leave it on — "
                                         "the point of this workflow is to change one "
                                         "setting and run again, and without this the "
                                         "second run continues from the first one's kernels "
                                         "instead of starting over."),
                # Appended, not inserted. A widget added anywhere but the end shifts every
                # value saved after it in every existing workflow, silently.
                io.String.Input("layer", default="",
                                tooltip="Leave empty for a one-layer bank.\n\nName a layer "
                                        "— conv2d_2, say — and only that layer is trained, "
                                        "judged by its own output rather than the model's. "
                                        "That is what lets a stack be taught one layer at a "
                                        "time: train the first, then chain a second Discover "
                                        "node naming the next one. The layer below keeps "
                                        "what it learned, and the new layer has to find "
                                        "structure in *its* responses — combinations of "
                                        "edges rather than edges.\n\nIt only means anything "
                                        "if there is a nonlinearity in between. Two "
                                        "convolutions with nothing between them collapse "
                                        "into one convolution."),
            ],
            outputs=[
                Model.Output(display_name="trained model"),
                History.Output(display_name="history"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, objective: str = "sparse response",
                diversity: float = 1.0, epochs: int = 120, batch_size: int = 512,
                learning_rate: float = 0.02, seed: int = 0, early_stopping: int = 0,
                live_preview: bool = True, reset_weights: bool = True,
                layer: str = "") -> io.NodeOutput:
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        if getattr(dataset, "task", "") != "discovery":
            raise NeurodesError(
                f"This node needs a dataset with no targets, but {dataset.name!r} is a "
                f"{dataset.task} dataset.",
                hint="Use the Image Patches node. Everything else in the pack has an answer "
                     "to copy, which is what makes this one different.",
            )
        cfg = T.TrainConfig(epochs=int(epochs), batch_size=int(batch_size),
                            learning_rate=float(learning_rate), loss=str(objective),
                            diversity=float(diversity), seed=int(seed),
                            layer=str(layer).strip(), reset_weights=bool(reset_weights),
                            early_stopping=int(early_stopping))

        total = max(1, int(cfg.epochs))
        pbar = ProgressBar(total)
        history = T.History()
        state = {"epoch": 0}

        # The diversity term estimates a correlation between every pair of kernels from one
        # batch. With fewer rows than pairs that estimate is mostly noise, and the term ends
        # up pushing the kernels around at random instead of apart. When a layer is named,
        # the count that matters is that layer's, not the model's last one.
        shape = DS.shape_at(model, cfg.layer)
        width = shape[1]
        kernels = width.size if width.is_concrete else 0
        rows = cfg.batch_size * (1 if shape.rank == 2 else 4)
        if diversity and kernels and rows < 4 * kernels:
            history.notes.append(
                f"A batch of {cfg.batch_size} is small for {kernels} kernels: the diversity "
                "term has to estimate a correlation for every pair of them from each batch. "
                f"Raise batch_size to at least {4 * kernels}, or use fewer kernels.")

        def on_progress(step: int, total_steps: int, info: dict) -> None:
            epoch = int(info.get("epoch", 0))
            if epoch != state["epoch"]:
                state["epoch"] = epoch
                pbar.update_absolute(epoch - 1, total,
                                     _preview(history) if live_preview else None)

        def should_stop() -> bool:
            mm.throw_exception_if_processing_interrupted()
            return False

        result = T.train(model, dataset, cfg, on_progress=on_progress,
                         should_stop=should_stop, history=history)
        pbar.update_absolute(total, total)

        scope = (f"layer {cfg.layer} of {model.model_name}" if cfg.layer
                 else model.model_name)
        report = (f"{scope}: {model.n_parameters():,} parameters in the model, "
                  f"{objective}, diversity {diversity}\n{result.summary()}\n\n"
                  + DS.report(model, dataset, float(diversity), layer=cfg.layer))
        return io.NodeOutput(model, result, report, ui=ui.PreviewText(report))


class NeuroEvaluate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroEvaluate",
            display_name="Evaluate",
            category=CAT,
            description="Scores a model on a dataset without training it. Use it on data the "
                        "model has never seen — a number from the training set only tells you "
                        "how well it memorised.",
            search_aliases=["test", "score", "accuracy", "validate", "metrics"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset"),
                io.Combo.Input("split", options=["validation", "train", "both"],
                               default="validation"),
            ],
            outputs=[
                io.String.Output(display_name="report"),
                io.Float.Output(display_name="accuracy"),
                io.Float.Output(display_name="loss"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, split: str = "validation") -> io.NodeOutput:
        loss_name = T.resolve_loss("auto", dataset)
        # Not _LOSS_FNS directly: an unsupervised objective is not in that table, and looking
        # it up there would fail with a bare KeyError naming nothing.
        loss_fn = T.make_loss(loss_name, dataset, T.TrainConfig())
        device = T.resolve_device("auto")
        model.to(device)
        bundle = dataset.to(device)

        lines, headline_acc, headline_loss = [], float("nan"), float("nan")
        for which in (["train", "validation"] if split == "both" else [split]):
            x = bundle.train_inputs if which == "train" else bundle.val_inputs
            y = bundle.y_train if which == "train" else bundle.y_val
            loss, acc = T.evaluate(model, x, y, loss_fn, loss_name, bundle.task)
            line = f"{which:<11} loss {loss:.4f}"
            if bundle.task == "classification":
                line += f"   accuracy {acc * 100:.2f}%"
            lines.append(line + f"   ({int(x[0].shape[0])} examples)")
            if which == split or split == "both" and which == "validation":
                headline_acc, headline_loss = acc, loss
        report = f"{model.model_name} on {bundle.name}\n" + "\n".join(lines)
        return io.NodeOutput(report, float(headline_acc), float(headline_loss),
                             ui=ui.PreviewText(report))


class NeuroPredictImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroPredictImages",
            display_name="Predict (Images)",
            category=CAT,
            description="Runs a trained model on a batch of ComfyUI images and reports what "
                        "it thinks they are. The bridge back out to the rest of ComfyUI: "
                        "anything you can load or generate, you can now classify.",
            search_aliases=["classify", "inference", "run model", "predict"],
            inputs=[
                Model.Input("model"),
                io.Image.Input("images"),
                io.String.Input("class_names", default="",
                                tooltip="Optional names, separated by commas, in class order."),
                io.Boolean.Input("channels_first", default=True, advanced=True,
                                 tooltip="Convert to [batch, channels, height, width]."),
                io.Boolean.Input("greyscale", default=False,
                                 tooltip="Average the colour channels first, for a model "
                                         "trained on one-channel images such as MNIST."),
            ],
            outputs=[
                io.String.Output(display_name="predictions"),
                io.Int.Output(display_name="top class"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, images, class_names: str = "", channels_first: bool = True,
                greyscale: bool = False) -> io.NodeOutput:
        x = images.float()
        if greyscale:
            x = x.mean(dim=-1, keepdim=True)
        if channels_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        expected = model.input_shapes[0]
        got = [int(s) for s in x.shape]
        want = [d.size if d.is_concrete else got[i] for i, d in enumerate(expected.dims)]
        if len(want) != len(got) or any(a != b for a, b in zip(want, got)):
            raise NeurodesError(
                f"The model expects {expected} but these images are "
                f"[{', '.join(str(g) for g in got)}].",
                hint="Resize the images to match, turn 'greyscale' on for a one-channel "
                     "model, or check the 'channels_first' setting.",
            )
        logits = T.predict(model, x)
        names = [p.strip() for p in str(class_names).split(",") if p.strip()]
        if logits.dim() == 1 or logits.shape[-1] == 1:
            scores = torch.sigmoid(logits.reshape(-1))
            top = (scores > 0.5).long()
            lines = [f"image {i}: {'yes' if int(t) else 'no'}  ({s * 100:.1f}%)"
                     for i, (t, s) in enumerate(zip(top.tolist(), scores.tolist()))]
        else:
            probs = torch.softmax(logits, dim=-1)
            top = probs.argmax(dim=-1)
            lines = []
            for i, (k, row) in enumerate(zip(top.tolist(), probs.tolist())):
                label = names[k] if k < len(names) else str(k)
                lines.append(f"image {i}: {label}  ({row[k] * 100:.1f}%)")
        text = "\n".join(lines)
        return io.NodeOutput(text, int(top[0].item()) if top.numel() else 0,
                             ui=ui.PreviewText(text))


class NeuroForwardImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroForwardImages",
            display_name="Forward (Images)",
            category=CAT,
            description="One forward pass. Pictures in, whatever the model makes of them "
                        "out, as an ordinary ComfyUI IMAGE.\n\nThis is inference: no "
                        "gradients, no training, nothing changes. Swap this in where Train "
                        "was and the same graph becomes a filter you can point at anything "
                        "— a photograph, a render, a frame of video.\n\nThe model has to "
                        "produce something image-shaped. For a classifier, use "
                        "Predict (Images) instead.",
            search_aliases=["inference", "run model", "apply", "forward pass", "infer",
                            "use model", "img2img", "filter", "predict image"],
            inputs=[
                Model.Input("model"),
                io.Image.Input("images"),
                io.Boolean.Input("greyscale", default=False,
                                 tooltip="Average the colour channels first, for a model "
                                         "trained on one-channel images."),
                io.Boolean.Input("resize", default=True,
                                 tooltip="Resize the input to the size the model declared. "
                                         "Turn it off to run a convolutional model at its "
                                         "native resolution — they do not mind."),
                *save_inputs("forward"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="stats"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, images, greyscale: bool = False, resize: bool = True,
                save: bool = False, filename_prefix: str = "neurodes/forward"
                ) -> io.NodeOutput:
        shape = model.input_shapes[0]
        with allocating():
            x = adopt(R.image_to_model_input(images, shape, greyscale=bool(greyscale),
                                             resize=bool(resize)))
            y = T.predict(model, x)
        if y.dim() != 4:
            raise NeurodesError(
                f"The model returned {tuple(y.shape)}, which is not a picture.",
                hint="Forward (Images) is for models that produce images — an autoencoder, "
                     "a U-Net, an upscaler. A classifier's scores go to Predict (Images), "
                     "and any intermediate tensor can be drawn with Capture Activations.",
            )
        out = R.model_input_to_image(y)
        return image_result(cls, out, R.describe(y), save=save,
                            filename_prefix=filename_prefix)


class NeuroSampleDiffusion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroSampleDiffusion",
            display_name="Sample (Diffusion)",
            category=CAT,
            description="Start from pure noise and let a trained diffusion model turn it "
                        "into a picture.\n\nThe loop is small enough to describe in a "
                        "sentence: ask the model what the noise is, take a little of it "
                        "away, and go round again. That is what a KSampler does, with the "
                        "same maths and a great deal more engineering.\n\nThe dataset input "
                        "is not for data — it carries the noise schedule the model was "
                        "trained with, so the two cannot drift apart.",
            search_aliases=["ksampler", "sample", "generate", "denoise loop", "ddim",
                            "diffusion", "txt2img", "reverse process"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset", tooltip="The Dataset As Diffusion node that trained "
                                                 "this model. Read for its noise schedule."),
                io.Int.Input("count", default=8, min=1, max=64,
                             tooltip="How many pictures to make."),
                io.Int.Input("steps", default=30, min=1, max=500,
                             tooltip="How many times round the loop. This sampler is "
                                     "deterministic, so more steps should not change the "
                                     "picture much — if it does, something is wrong."),
                io.Int.Input("seed", default=0, min=0, max=1 << 31,
                             control_after_generate=True,
                             tooltip="Which noise to start from. This is the only thing that "
                                     "decides what you get."),
                io.Float.Input("guidance", default=0.0, min=0.0, max=3.0, step=0.05,
                               tooltip="Exaggerate what makes each picture differ from the "
                                       "batch average. Not classifier-free guidance — there "
                                       "is nothing to condition on — but the same trick, and "
                                       "it sharpens an undertrained model. 0 is honest."),
                io.Int.Input("size", default=0, min=0, max=512, advanced=True,
                             tooltip="Make pictures at a different size than it trained on. "
                                     "0 uses the training size. A convolutional model will "
                                     "happily go bigger; it just tiles what it knows."),
                *save_inputs("sample"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Image.Output(display_name="reverse process",
                                tooltip="One frame per step, the whole batch tiled — an "
                                        "image sequence, so a video node turns it into a "
                                        "clip of the picture appearing out of the noise."),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, count: int = 8, steps: int = 30, seed: int = 0,
                guidance: float = 0.0, size: int = 0, save: bool = False,
                filename_prefix: str = "neurodes/sample") -> io.NodeOutput:
        cfg = DF.config_of(dataset)
        side = int(size) or 0
        trained = tuple(cfg.get("size", (0, 0)))
        try:
            with allocating():
                final, trajectory = DF.sample(
                    model, cfg, count=int(count), steps=int(steps), seed=int(seed),
                    size=(side, side) if side else None, guidance=float(guidance),
                    keep_trajectory=True)
        except RuntimeError as exc:
            if not side:
                raise
            # A U-Net halves and doubles, so a size that does not divide cleanly comes back
            # from the upsampling a pixel short and the Concat refuses it. Torch says this
            # in terms of tensor dimensions; say it in terms of the widget that caused it.
            raise NeurodesError(
                f"The model could not run at {side} x {side}.",
                hint=f"Each Max Pool halves the picture and each Upsample doubles it, so the "
                     f"size has to survive the round trip — with two of each, a multiple of "
                     f"4. Try {max(4, round(side / 4) * 4)}, or set size back to 0 to use the "
                     f"{trained[0]} x {trained[1]} it trained on.\n\n{exc}",
            ) from exc
        images = R.model_input_to_image(final)
        frames = R.model_input_to_image(trajectory)
        per_step = frames.shape[0] // max(1, int(steps))
        film = torch.stack([R.tile_batch(frames[i * per_step:(i + 1) * per_step])
                            for i in range(int(steps))]) if per_step else images
        report = (f"{model.model_name}: {count} sample(s) in {steps} steps\n"
                  f"  schedule    {cfg.get('schedule')}, predicting the {cfg.get('predict')}\n"
                  f"  size        {images.shape[1]} x {images.shape[2]}\n"
                  f"  " + R.describe(final))
        return image_result(cls, images, film, report, save=save,
                            filename_prefix=filename_prefix)


class NeuroGenerateText(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NeuroGenerateText",
            display_name="Generate (Text)",
            category=CAT,
            description="Let a trained language model write.\n\nThe loop is the one behind "
                        "every language model: ask for the distribution over the next "
                        "character, draw one from it, stick it on the end and ask again. "
                        "That is all autoregressive generation is.\n\nThe dataset input is "
                        "not for data — it carries the vocabulary, which is the only way to "
                        "turn the model's numbers back into letters.",
            search_aliases=["gpt", "sample text", "language model", "write", "complete",
                            "autoregressive", "llm", "nanogpt"],
            inputs=[
                Model.Input("model"),
                Dataset.Input("dataset", tooltip="The Text Dataset this model trained on."),
                io.String.Input("prompt", default="", multiline=True,
                                tooltip="Text to start from. Empty starts from one random "
                                        "character. Characters the model has never seen are "
                                        "dropped, since it has no number for them."),
                io.Int.Input("length", default=400, min=1, max=8000,
                             tooltip="How many characters to write."),
                io.Float.Input("temperature", default=0.8, min=0.05, max=2.0, step=0.05,
                               tooltip="How boldly to draw. Near 0 it always takes the most "
                                       "likely character and gets stuck repeating; above "
                                       "about 1.2 it comes apart into noise."),
                io.Int.Input("top_k", default=0, min=0, max=512,
                             tooltip="Only ever draw from the k most likely characters. 0 "
                                     "considers all of them. A small k tidies up an "
                                     "undertrained model by hiding its worst guesses.",
                             advanced=True),
                io.Int.Input("seed", default=0, min=0, max=1 << 31,
                             control_after_generate=True),
            ],
            outputs=[io.String.Output(display_name="text")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, dataset, prompt: str = "", length: int = 400,
                temperature: float = 0.8, top_k: int = 0, seed: int = 0) -> io.NodeOutput:
        cfg = TX.config_of(dataset)
        with allocating():
            written = TX.generate(model, cfg, prompt=str(prompt), length=int(length),
                                  temperature=float(temperature), top_k=int(top_k),
                                  seed=int(seed))
        return io.NodeOutput(written, ui=ui.PreviewText(written))


TRAIN_NODES = [NeuroTrainer, NeuroTrain, NeuroDiscover, NeuroEvaluate, NeuroPredictImages,
               NeuroForwardImages, NeuroSampleDiffusion, NeuroGenerateText]
