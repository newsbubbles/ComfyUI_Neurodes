# Research notes — ComfyUI custom node API (for neurodes)

Ground truth: the local install at `C:\Users\dumbass\ComfyUI`, **ComfyUI 0.17.0**, frontend
package **1.41.20**. Everything below was read out of that source tree, not from docs.
Line references are to that version.

---

## 1. The V3 node schema is the target

`comfy_api/latest/_io.py` (85 KB) is the public node API. A V3 node is:

```python
from comfy_api.latest import io, ui, ComfyExtension

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="...", display_name="...", category="...",
                         inputs=[...], outputs=[...])

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(value, ui=ui.PreviewText("..."))
```

Nodes are **stateless classmethods** (`_io.py:1997`). No `self`, no instance state. Optional
classmethod hooks: `validate_inputs`, `fingerprint_inputs` (= V1 `IS_CHANGED`),
`check_lazy_status`.

`io.Schema` fields that matter to us (`_io.py:1430`):

| field | use in neurodes |
|---|---|
| `node_id` | globally unique; we prefix everything `Neuro*` |
| `category` | `"neurodes/..."` drives the Add-Node menu tree |
| `search_aliases` | synonyms — `Dense` finds `Linear`, `FC` finds `Linear` |
| `is_output_node` | forces execution even with nothing downstream (plots, exports) |
| `not_idempotent` | rerun even when inputs are unchanged |
| `is_experimental` | greys the node; we skip it |
| `description` | hover tooltip; our teaching surface |

### Registration

`nodes.py:2248-2295`. The loader checks, in this order:

1. `WEB_DIRECTORY` on the module → registers a static web dir. **Checked before** the
   V1/V3 branch, so it coexists with a V3 entrypoint.
2. `NODE_CLASS_MAPPINGS` → V1 path.
3. `elif comfy_entrypoint` → V3 path. Must be callable (sync or async), must return a
   `ComfyExtension`, whose `get_node_list()` returns `list[type[io.ComfyNode]]`.

Because it is `elif`, **do not define both** `NODE_CLASS_MAPPINGS` and `comfy_entrypoint` —
V1 wins and the V3 nodes silently vanish.

`ComfyExtension` (`comfy_api/latest/__init__.py:118`) also has an `async on_load()` for
one-time init.

Failure mode to know: exceptions inside `comfy_entrypoint` are swallowed into a
`logging.warning` (`nodes.py:2291`) and the pack just doesn't appear. Any import-time cost
or error must be avoided — hence lazy torch import.

## 2. Custom socket types are one line

```python
io.Custom("NEURO_TENSOR").Input("x")
io.Custom("NEURO_TENSOR").Output()
```

`Custom()` (`_io.py:132`) builds a `ComfyTypeIO` subclass bound to an arbitrary io_type
string. The frontend assigns a link colour per type name automatically and refuses
mismatched connections. **This is the entire reason the project works**: the connection
rules for a neural network are already enforceable by ComfyUI's own type system.

ComfyUI's own training nodes use exactly this: `io.Custom("LOSS_MAP")` in
`comfy_extras/nodes_train.py:1372`.

## 3. Dynamic inputs — three mechanisms, all first class

### `io.DynamicCombo` (`_io.py:1091`)

A combo widget where **the selected option decides which further inputs appear on the node**.

```python
io.DynamicCombo.Input("kind", options=[
    io.DynamicCombo.Option("linear", [io.Int.Input("out_features", default=128)]),
    io.DynamicCombo.Option("conv2d", [io.Int.Input("out_channels"), io.Int.Input("kernel")]),
])
```

The value arrives in `execute` as a **flat dict**: `kind["kind"]` is the selected key and
`kind["out_features"]` is the nested value (confirmed by `nodes_glsl.py:828`,
`size_mode["size_mode"] == "custom"` then `size_mode["width"]`).

This is how a single "Layer" node can morph its parameter set. Real precedent in core:
`comfy_extras/nodes_glsl.py:772`.

### `io.Autogrow` (`_io.py:953`)

Grows a slot as you connect to it. Two templates:

- `TemplatePrefix(input, prefix="image", min=1, max=32)` → `image0, image1, …`
- `TemplateNames(input, names=[...], min=…)` → fixed name list

Value arrives as `dict[str, Any]`; iterate `.values()`, order preserved, unconnected slots
are `None`.

**Gotcha that decided our Shape design:** `_AutogrowTemplate.__init__` (`_io.py:963`) does
`self.input.force_input = True` for any `WidgetInput`. So an autogrow of `io.Int.Input`
gives you N *sockets*, never N spinboxes. Autogrow is right for "N tensors into a Concat",
wrong for "type the 4 numbers of a shape".

### `io.MatchType` (`_io.py:877`)

A generic slot. Inputs and outputs sharing a `MatchType.Template("name")` resolve to
whatever concrete type gets connected, with optional `allowed_types`. Core uses it for
`Switch` (`comfy_extras/nodes_logic.py:14`). Useful for a passthrough/reroute that keeps
the neurodes type.

Also present: `io.MultiType.Input(id, types=[...])` for "accepts A or B", and
`io.DynamicSlot` for a socket that reveals sub-inputs when connected.

## 4. UI feedback from a node

`comfy_api/latest/_ui.py`: `PreviewText` (:455), `PreviewImage` (:389), `PreviewMask`,
`PreviewAudio`, `PreviewVideo`, `PreviewUI3D`, `SavedImages`.

Returned via `io.NodeOutput(value, ui=ui.PreviewText("[B, 128]"))`.

Progress during a long node:

```python
from comfy.utils import ProgressBar          # nodes_train.py:24
pbar = ProgressBar(total_steps); pbar.update(1)
```

and the newer `ComfyAPI.Execution.set_progress(value, max_value, preview_image=…)`
(`comfy_api/latest/__init__.py:37`) which can push a **live preview image** each step — a
live-updating loss curve during training is therefore possible.

Interruption: `comfy.model_management.throw_exception_if_processing_interrupted()` inside
the training loop, so Cancel works.

## 5. Environment facts

ComfyUI's own venv (`C:\Users\dumbass\ComfyUI\venv`, py3.11.9):

- torch 2.10.0+cu126, torchvision 0.25.0+cu126
- numpy **1.26.4** (not 2.x — avoid numpy-2-only APIs)
- PIL 12.1.1, matplotlib 3.10.8, scipy 1.17.1

System python 3.11.9 has torch 2.11.0+cu126, CUDA available.

**Dependency decision:** ComfyUI hard-requires torch, numpy and PIL. matplotlib and
torchvision are present here but are *not* guaranteed on a user's install (matplotlib is
almost certainly pulled in by one of the 22 other custom node packs). So neurodes ships
with **zero new requirements**: plotting is hand-rolled on PIL, and torchvision datasets
are an optional import guarded at call time.

Also relevant: this machine's GTX 1080 thermally throttles to ~7% clock, so all defaults
must be CPU-sane and every example must train in seconds, not minutes.

## 6. Consequences for neurodes

1. Node graph → symbolic trace, because a V3 `execute` is a pure classmethod with cached
   outputs. Real `nn.Module`s must be built *once*, at the Model node, not per layer node.
2. Shape errors can be raised as plain exceptions in `execute`; ComfyUI surfaces them on
   the offending node. Message quality is the whole teaching experience.
3. Layer parameter sets differ wildly between layer types → either N discrete nodes or one
   `DynamicCombo`. We do both from one registry.
4. `WEB_DIRECTORY` lets us ship JS, so inferred shapes can be drawn onto the node body
   after a run instead of only living in a text preview.
5. Rerunning training with unchanged inputs would hit the node cache → the training node
   needs a `seed` with `control_after_generate`, the ComfyUI-native way to say "go again".
