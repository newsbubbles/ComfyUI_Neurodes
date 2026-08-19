"""Turning a trace into a real ``nn.Module``.

This is the one place where the architecture stops being a description and becomes
something with weights. Everything upstream of here is cheap and repeatable; everything
downstream is training.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from .errors import BuildError, GraphError, ShapeError
from .registry import LayerSpec, get
from .runtime import allocating, trainable
from .shape import Shape
from .trace import Op, SymTensor, assign_names, graph_inputs, share_groups, topo_ops


class _Step:
    """One op, ready to run: which module, which config, where its inputs come from."""

    __slots__ = ("op", "name", "spec", "cfg", "sources")

    def __init__(self, op: Op, name: str, spec: LayerSpec, cfg: dict, sources: list[int]):
        self.op, self.name, self.spec, self.cfg, self.sources = op, name, spec, cfg, sources


class CompiledModel(nn.Module):
    """A traced neurodes graph, built into a runnable torch model.

    ``forward`` takes one tensor per Input node, in the order those Input nodes are reached
    from the outputs, and returns one tensor per model output (or a tuple when there are
    several).
    """

    def __init__(self, outputs: Sequence[SymTensor], name: str = "Model"):
        super().__init__()
        if not outputs:
            raise GraphError(
                "A model needs at least one output.",
                hint="Connect a tensor into the Build Model node's output slot.",
            )
        self.model_name = name
        self.outputs = list(outputs)
        self.ops = topo_ops(self.outputs)
        self.input_ops = graph_inputs(self.outputs)
        if not self.input_ops:
            raise GraphError(
                "This model has no Input node.",
                hint="Every network starts with an Input node, which is where you declare "
                     "the shape of the data going in.",
            )
        self.names = assign_names(self.ops)
        share_groups(self.ops)

        self._modules_by_name = nn.ModuleDict()
        self.plan: list[_Step] = []
        self._input_index = {op.uid: i for i, op in enumerate(self.input_ops)}

        # Weights are allocated here. They must not be inference tensors: see runtime.py.
        with trainable():
            for op in self.ops:
                if op.kind == "input":
                    continue
                spec = get(op.kind)
                shapes = [t.shape for t in op.inputs]
                name = self.names[op.uid]
                try:
                    cfg = spec.runtime_cfg(shapes, op.params)
                except ShapeError as exc:
                    raise BuildError(f"{spec.display} ({name}): {exc.message}", exc.hint) from None
                if name not in self._modules_by_name:
                    try:
                        module = spec.make_module(shapes, op.params)
                    except ShapeError as exc:
                        raise BuildError(f"{spec.display} ({name}): {exc.message}",
                                         exc.hint) from None
                    except Exception as exc:  # torch raised something we did not anticipate
                        raise BuildError(
                            f"Could not build {spec.display} ({name}): {exc}",
                            hint="This is usually a setting torch rejects. Check the node's "
                                 "values.",
                        ) from None
                    if module is not None:
                        self._modules_by_name[name] = module
                self.plan.append(_Step(op, name, spec, cfg,
                                       [t.producer.uid for t in op.inputs]))

        self.output_uids = [t.producer.uid for t in self.outputs]

        # A shared layer is one module but two call sites, and the two produce different
        # activations. Names must therefore be per-call, not per-module.
        self.step_names: dict[int, str] = {}
        used: dict[str, int] = {}
        for step in self.plan:
            used[step.name] = used.get(step.name, 0) + 1
            n = used[step.name]
            self.step_names[step.op.uid] = step.name if n == 1 else f"{step.name}#{n}"

    # -- running ------------------------------------------------------------
    def forward(self, *inputs):
        env, _ = self._run(self.plan, inputs)
        results = [env[uid] for uid in self.output_uids]
        return results[0] if len(results) == 1 else tuple(results)

    # -- description --------------------------------------------------------
    @property
    def input_shapes(self) -> list[Shape]:
        return [t.shape for t in _input_tensors(self.outputs, self.input_ops)]

    @property
    def output_shapes(self) -> list[Shape]:
        return [t.shape for t in self.outputs]

    def n_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def reinitialise(self, seed: int | None = None) -> None:
        """Throw the weights away and draw new ones.

        Needed because ComfyUI caches a node's output: a Build Model node whose widgets have
        not changed is not re-executed, so the *same model object* is handed to the trainer
        on every run. Without this, pressing run a second time silently continues from the
        weights the first run left behind — so two runs of a workflow do not compare two
        settings, they compare one setting against itself plus more training.

        Every parameter in the pack lives inside a torch module that knows how to
        re-initialise itself. Buffers, which is where a fixed positional encoding lives, are
        deliberately untouched: they were never learned.

        A shared layer is one module used at two call sites, and ``modules()`` yields each
        object once, so tied weights stay tied.
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset) and module is not self:
                reset()

    @property
    def device(self) -> torch.device:
        """Where this model's weights currently live.

        Training moves the model to the GPU, so anything handed to it afterwards has to
        follow it there. A model with no parameters at all lives on the CPU.
        """
        for parameter in self.parameters():
            return parameter.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    def _module(self, name: str):
        """``nn.ModuleDict`` has no ``.get``, and stateless layers have no module at all."""
        return self._modules_by_name[name] if name in self._modules_by_name else None

    def module_for(self, op: Op):
        return self._module(self.names.get(op.uid, ""))

    # -- looking inside ------------------------------------------------------
    @property
    def layer_names(self) -> list[str]:
        """Every named point in the network, in execution order."""
        seen: set[int] = set()
        out: list[str] = []
        for op in self.input_ops:
            if op.uid not in seen:
                seen.add(op.uid)
                out.append(self.names[op.uid])
        for step in self.plan:
            if step.op.uid not in seen:
                seen.add(step.op.uid)
                out.append(self.step_names[step.op.uid])
        return out

    def forward_capturing(self, *inputs) -> tuple[Any, dict[str, torch.Tensor]]:
        """A normal forward pass that also hands back every intermediate tensor.

        In ordinary PyTorch this needs a forward hook on every module. Here the forward
        pass already keeps each intermediate in ``env`` in order to feed the next step, so
        capturing them is just a matter of not throwing them away.
        """
        env, captured = self._run(self.plan, inputs)
        for step in self.plan:
            captured[self.step_names[step.op.uid]] = env[step.op.uid]
        results = [env[uid] for uid in self.output_uids]
        return (results[0] if len(results) == 1 else tuple(results)), captured

    def forward_to(self, layer: str, *inputs) -> torch.Tensor:
        """Run only the part of the network that ``layer`` depends on.

        This is what lets deep dream work on a canvas far bigger than the model ever saw:
        the convolutional prefix of a network does not care about resolution, and stopping
        before the Flatten means the size-fixed part is never reached.
        """
        target = self._step_named(layer)
        needed = self._dependencies(target.op.uid)
        partial = [s for s in self.plan if s.op.uid in needed]
        env, _ = self._run(partial, inputs)
        return env[target.op.uid]

    def _run(self, steps, inputs) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
        if len(inputs) != len(self.input_ops):
            raise BuildError(
                f"This model takes {len(self.input_ops)} input tensor(s), got {len(inputs)}.",
                hint="Inputs are passed in the order their Input nodes appear in the graph: "
                     + ", ".join(op.params.get("name", "x") for op in self.input_ops),
            )
        env: dict[int, torch.Tensor] = {}
        named: dict[str, torch.Tensor] = {}
        for op, tensor in zip(self.input_ops, inputs):
            env[op.uid] = tensor
            named[self.names[op.uid]] = tensor
        for step in steps:
            args = [env[uid] for uid in step.sources]
            env[step.op.uid] = step.spec.run(self._module(step.name), args, step.cfg)
        return env, named

    def _step_named(self, layer: str) -> _Step:
        wanted = (layer or "").strip()
        for step in self.plan:
            if self.step_names[step.op.uid] == wanted:
                return step
        raise BuildError(
            f"This model has no layer called {wanted!r}.",
            hint="Available: " + ", ".join(self.layer_names),
        )

    def _dependencies(self, target_uid: int) -> set[int]:
        by_uid = {s.op.uid: s for s in self.plan}
        keep: set[int] = set()
        stack = [target_uid]
        while stack:
            uid = stack.pop()
            if uid in keep:
                continue
            keep.add(uid)
            step = by_uid.get(uid)
            if step is not None:
                stack.extend(step.sources)
        return keep

    def dummy_inputs(self, batch: int = 2, device=None) -> list[torch.Tensor]:
        """Fabricate one plausible tensor per Input, for a smoke-test forward pass."""
        made = []
        with allocating():
            for t in _input_tensors(self.outputs, self.input_ops):
                sizes = t.shape.concrete_or_placeholder(batch)
                if t.dtype == "int64":
                    high = _embedding_limit(self, t) or 2
                    made.append(torch.randint(0, max(high, 1), sizes, device=device))
                elif t.dtype == "bool":
                    made.append(torch.zeros(sizes, dtype=torch.bool, device=device))
                else:
                    made.append(torch.randn(sizes, device=device, dtype=_torch_float(t.dtype)))
        return made

    def verify(self, batch: int = 2) -> list[Shape]:
        """Run a dummy batch and check reality against what the graph claimed.

        Any disagreement is a bug in a layer's ``infer``, and it is far better to find it
        here than after twenty minutes of training.
        """
        self.eval()
        with torch.no_grad():
            result = self(*self.dummy_inputs(batch))
        actual = [result] if isinstance(result, torch.Tensor) else list(result)
        real = [Shape([int(s) for s in t.shape]) for t in actual]
        for claimed, got in zip(self.output_shapes, real):
            expected = claimed.concrete_or_placeholder(batch)
            if tuple(int(d.size) for d in got.dims) != tuple(expected):
                raise BuildError(
                    f"Shape inference disagrees with the real model: the graph says "
                    f"{claimed} but running it gave {got}.",
                    hint="This is a bug in neurodes, not in your workflow.",
                )
        return real


def _torch_float(name: str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        name, torch.float32)


def _input_tensors(outputs: Sequence[SymTensor], input_ops: Sequence[Op]) -> list[SymTensor]:
    """Recover the SymTensor each Input op produced, by walking the graph."""
    found: dict[int, SymTensor] = {}
    stack = list(outputs)
    seen: set[int] = set()
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if t.producer is None:
            continue
        if t.producer.kind == "input":
            found[t.producer.uid] = t
        stack.extend(t.producer.inputs)
    missing = [op for op in input_ops if op.uid not in found]
    if missing:
        raise GraphError("Could not resolve every Input tensor while building the model.")
    return [found[op.uid] for op in input_ops]


def _embedding_limit(model: CompiledModel, tensor: SymTensor) -> int:
    """Largest safe index for an int input, so a dummy batch does not blow up an Embedding."""
    limits = [int(step.op.params.get("num_embeddings", 0))
              for step in model.plan if step.op.kind == "embedding"]
    return min(limits) if limits else 0


def build_model(outputs: Sequence[SymTensor], name: str = "Model",
                verify: bool = True) -> CompiledModel:
    """Compile a trace, and by default prove it runs before handing it back."""
    model = CompiledModel(outputs, name=name)
    if verify:
        try:
            model.verify()
        except BuildError:
            raise
        except Exception as exc:
            raise BuildError(
                f"The model was built but a test batch failed to pass through it: {exc}",
                hint="The shapes line up on paper. This usually means a setting torch itself "
                     "rejects, such as an index outside an Embedding's range.",
            ) from None
    return model


def parameter_count(op: Op) -> int:
    """How many weights one op would own, without allocating any memory for them.

    Uses torch's meta device, so asking about a 4096x4096 Linear costs nothing.
    """
    spec = get(op.kind)
    if spec.build is None:
        return 0
    shapes = [t.shape for t in op.inputs]
    try:
        with torch.device("meta"):
            module = spec.make_module(shapes, op.params)
    except Exception:
        try:
            module = spec.make_module(shapes, op.params)
        except Exception:
            return 0
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())
