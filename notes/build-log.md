# Build log

A running record of decisions, bugs and things that turned out not to be true. Kept because
the *reasons* are the part that gets lost, and because several of these cost real time and
would cost it again.

Companion documents: `design.md` (why the architecture is what it is) and `research.md`
(what the ComfyUI V3 API actually does, read out of the source).

---

## 1. The founding decision

A ComfyUI graph is the Keras Functional API. `h = Dense(128)(x)` read vertically is a node
with a tensor in and a tensor out, so the mapping needs no invention.

The consequence that everything else follows from: **the wire carries a symbolic tensor**
(shape + dtype + back-pointer to its producer), and running the graph *traces* the network
rather than executing it. Forced by the runtime — a V3 `execute` is a cached, stateless
classmethod, so real `nn.Module`s built per node would have chaotic parameter lifetimes —
but better anyway: editing stays instant, shapes are known at design time, and errors land
on the node that caused them.

Alternatives considered and rejected are written up in `design.md §4`.

## 2. One registry, four consumers

Each layer is declared once in `core/registry.py` with `infer` / `build` / `emit_init` /
`emit_call` / `params`. From that single table we generate the discrete nodes, the `Layer`
dropdown, the code exporter and the summary table.

This is the load-bearing structural choice. It means the shape a node *shows* you and the
module that gets *built* cannot disagree, because they are the same declaration. Adding a
layer is one entry and no node code.

## 3. Bugs worth remembering

### 3.1 ComfyUI runs prompts inside `torch.inference_mode()`

**Symptom:** `RuntimeError: element 0 of tensors does not require grad and does not have a
grad_fn`, at the first `loss.backward()`. Only inside ComfyUI; headless tests passed.

**Cause:** inference mode is stronger than `no_grad`. A tensor *created* while it is active
is permanently barred from autograd. So `torch.enable_grad()` alone does nothing — the
weights were already poisoned in an earlier node.

**Fix:** `core/runtime.py`. Three separate places have to escape it: building the model,
building the dataset, and training. Tensors handed in by the host (a ComfyUI IMAGE used as
training data) must be cloned out with `adopt()`.

ComfyUI's own `comfy_extras/nodes_train.py` does the same thing, which confirmed it was the
intended escape hatch rather than a hack.

### 3.2 Cached node outputs are shared mutable objects

**Symptom:** run 1 fine, run 2 fails in **Train** with the same grad error as 3.1 — which
sent the diagnosis down the wrong path at first.

**Cause:** `dream()` did `parameter.requires_grad_(False)`. ComfyUI caches Build Model, so
the next run's Train got the same object with its gradients switched off.

**Fix:** delete the line. `torch.autograd.grad(loss, x)` returns only the gradient with
respect to `x` and never accumulates into parameters, so it bought nothing.

**General rule:** treat anything arriving on an input socket as read-only. The symptom
appears in a *different node* than the cause, so suspect this whenever the first run works
and the second does not. Regression test must run the mutating node and then re-run the
upstream one — a single pass will never catch it.

### 3.3 The folder loader never noticed new images

**Symptom:** added images to the training folder, dataset size unchanged.

**Cause:** ComfyUI caches on input values, and widget values do not change when the folder
contents do.

**Fix:** `fingerprint_inputs` signing the folder's file list and modification times.

Found only because a test run reported an identical parameter count to the previous one.
Worth watching for: identical numbers across runs are suspicious.

### 3.4 Device placement

**Symptom:** `Expected all tensors to be on the same device` in Deep Dream, and later in
`evaluate`.

**Cause:** training moves the model to the GPU; anything handed to it afterwards has to
follow. CPU-only tests cannot catch this.

**Fix:** `CompiledModel.device`, used by the capture, dream and evaluate paths. The test for
it is gated on `torch.cuda.is_available()` and says so out loud when it skips, rather than
passing quietly on a machine that cannot exercise it.

### 3.5 A new widget silently rewrote every saved workflow

**Symptom:** after adding `early_stopping` to the Trainer, the bundled examples loaded with
`learning_rate` of `NaN`, `batch_size` of `0` and `shuffle` set to `auto`. No error anywhere.

**Cause:** a workflow stores widget values as a **positional array**. `early_stopping` went in
at index 1, so every value after it in the seven workflows saved beforehand shifted along by
one. The frontend fills in missing *trailing* values from the defaults, which is why adding a
widget at the end is harmless and adding one in the middle is not.

**Fix:** patch the seven files, and add a check that type-checks each saved value against the
schema slot it lands in — INT/FLOAT/BOOLEAN by type, COMBO by membership. A shift almost
always lands a float on an int or a string on a boolean, so this catches it. Trailing
omissions still pass, because those are legitimate.

Two false alarms the check had to be taught about, both real behaviour rather than drift:

- a **seed** widget is followed by a hidden `control_after_generate` value. It appears
  whenever the id is `seed`, whether or not the schema declares the flag — four data nodes
  were relying on that, and now say so explicitly.
- a `MultiType` input reports `"STRING,NEURO_SHAPE"`; the widget is the first entry. That is
  how `Input.shape` keeps a typed value while also accepting a connection.

**General rule:** adding a widget anywhere but the end is a breaking change to every saved
workflow. If it has to go in the middle, expect to migrate the files.

### 3.6 The summary described a model that had been thrown away

Not a crash — a false report. With early stopping restoring the best weights, the run summary
still printed `val acc 92.4% -> 97.6%` from the *last* epoch, while the weights in hand were
epoch 11's. `History.kept` now indexes the epoch the model actually holds, and every `->` in
the summary reads from it.

Worth stating as a rule: any figure printed after a restore has to be re-derived, not carried
over. The overfitting diagnosis is the exception — it is looking at the shape of the whole
curve on purpose, and now says out loud that early stopping already dealt with it.

### 3.7 The sampler got worse the harder it worked

**Symptom:** the tiny diffusion model produced recognisable shapes at 20 sampling steps and
pure static at 60. More effort, worse result.

**Cause:** each DDIM step splits the picture it holds into a clean guess and a noise term,
and re-mixes them at the next timestep. Near `t=1` the clean guess is divided by `sqrt(a)`
with `a ≈ 1e-4`, so the model's error is multiplied by a hundred and the guess lands far
outside `[0, 1]`. Clipping it back is correct and necessary — but the noise term was still
the model's raw output, which corresponds to the *unclipped* guess. The two halves of the
update described different pictures. Each step's disagreement is small; sixty of them are
not.

**Fix:** clip the guess, then rebuild the noise from the clipped guess, so
`x == sqrt(a)·clean + sqrt(1-a)·noise` holds going into the re-mix.

**The interesting part is the test.** The obvious one — sample at 8 steps and at 64 and
assert they agree — passes on the broken code, because both versions finish on a clipped
guess and the final images look alike. Reverting the fix and watching the test still pass
was the only reason I found that out. What does catch it is asserting the invariant
directly: `step()` was pulled out as its own function so the check can take its `clean` and
`noise`, put them back together at the timestep it was given, and demand the original
picture. That fails by 7.6 on the broken version.

**Rule:** a regression test written from the *symptom* can easily miss the *cause*. Break
the fix on purpose and confirm the test goes red, or you have not written a regression test.

### 3.8 Good advice, given to the wrong model

The trainer tells reconstruction models to put a Sigmoid on the end, since pictures live in
`[0, 1]`. A diffusion model predicts **noise**, which is negative half the time, and a
Sigmoid there would make half its answers unreachable. The check now looks at the range of
the *target* rather than the input.

Same shape of mistake in `dataset_preview`, which decided channels-first by testing
`shape[1] in (1, 3)` — a diffusion input has five channels, so it fell through and tried to
read a 32-pixel axis as colour.

Both are the same lesson: a rule inferred from every dataset that existed at the time.

### 3.9 Encoding

Two source files were corrupted by editing them through PowerShell's
`Get-Content -Raw | Set-Content -Encoding utf8`, which reads as ANSI and writes as UTF-8,
double-encoding every em-dash. **Never edit source through that pipeline.** Use the editor
tooling.

## 4. Things that were wrong on the first attempt

**Global pooling in the classifier examples.** Twice — MNIST at 61%, then the shape
classifier at 63%. Global average pooling collapses every spatial dimension, and for
"is this a circle or a square" the spatial arrangement *is* the signal. Flatten head:
97.7% and 94.2%. Worth remembering as a reflex: global pooling is for when you want
translation invariance, not when position matters.

**Deep dream produced corduroy.** Uniform diagonal hatching at any setting. Two causes,
both fixed:

- Gradient normalised by standard deviation. The original uses mean absolute value, which
  is far less twitchy when most of the gradient is zero — which after a ReLU it usually is.
- No gradient smoothing. Left alone, the optimiser finds that the cheapest way to excite an
  edge detector is a one-pixel grating. Blurring the *gradient* (not the picture) removes
  that shortcut. This is the single biggest difference between output that looks like
  something and output that looks like fabric.

Settled by sweeping into a contact sheet and reading the grid as one image, with a
"% at the rails" figure per tile to make saturation measurable rather than a matter of
opinion. Result: `strength` 0.006, `feature_scale` 0.9. `strength` turned out to be a
contrast dial — 0.002 gives 1% saturated, 0.04 gives 89%.

Also learned: canvas size does **not** change feature size. That is set by the receptive
field, so a bigger canvas simply fits more of them. Said out loud in the node tooltip
because it is the first thing anyone tries.

**Connecting optional inputs by slot index.** An optional socket shifts the numbering, so
`ds.connect(0, cap, 1)` put a dataset into the `images` slot and silently connected nothing.
Always connect by name when building graphs programmatically.

**The U-Net was laid out as one long row.** Twenty-five nodes in a line, with the skip
connections running the whole length of the graph and crossing everything on the way. It ran
correctly and taught nothing. Placed on three levels — encoder descending, bottleneck at the
bottom, decoder ascending — each skip becomes a horizontal wire at its own resolution, which
is exactly how the architecture is drawn on paper. The layout is part of the example, not
decoration: the whole premise of the pack is that an architecture is legible as a picture.

## 4b. Diffusion, and where the timestep goes

The one thing a diffusion model needs that this pack could not express: it takes **two**
inputs, a noisy picture and a timestep. `DataBundle` holds one `x` and one `y`, and `Train`
hands a model a single tensor. Supporting multiple inputs properly would ripple through
training, evaluation, the plots and the compatibility checks.

The timestep rides as **extra input channels** instead — a couple of planes holding a
sine/cosine encoding of `t`, concatenated onto the picture, so a 3-channel image becomes a
5-channel input. It is a technique real small diffusion models use rather than a dodge, it
needs no change to the trainer at all, and it is *better here*: the timestep is visible,
arriving on the wire, instead of appearing by magic inside every residual block. The node
tooltip suggests setting `time_channels` to 0 and watching the model produce mud, which is
the fastest way to understand why it needs to be there.

The encoding is chosen so `t` can be recovered from the planes — `atan2(sin, cos)/π` — which
is asserted in the tests and is also what lets the test suite build a perfect oracle model
for a one-picture world and demand the sampler reproduce that picture exactly.

Multi-input models are still the right thing eventually. This was the version that could ship
without destabilising everything else.

## 4c. And then multi-input models, properly

The gap closed. `DataBundle` grew `side_train` / `side_val` — further inputs after the first
— rather than turning `x_train` into a list, which would have broken every call site in the
pack for no gain. `x_train` is still the first input; `train_inputs` / `val_inputs` return
the whole tuple; single-input datasets have an empty one and behave exactly as before.

The blast radius was smaller than expected and entirely in the *consumers*: batching in
`_fit`, `evaluate`, `predict`, the confusion matrix, the reconstruction grid, and the
inspector's `_source_tensor`. Each was reaching into `data.x_val` directly and calling
`model(x)`. That is the tell: five places had independently assumed one input, and none of
them said so.

Two things it unlocked immediately:

- **The Siamese example trains.** It was architecture-only before, because there was no way
  to feed two towers. `Dataset As Pairs` draws matched and mismatched pairs from any labelled
  dataset, and the example goes from a picture of a network to 98.7% on same-or-different
  with 658 parameters — one tower's worth, used twice.
- **The diffusion timestep can be a real second input** rather than extra channels, which is
  what an actual implementation does: a small vector, projected by a Linear and added onto a
  feature map. Both modes are kept, because the channel version needs nothing of the model
  and is the easier first thing to understand.

`Forward (Images)` deliberately does *not* support multi-input models — an image batch is
one input and cannot supply the rest — and says so rather than failing in torch.

## 5. Where the epoch count went

Observed while watching real runs: the loss plateaus long before the epoch count runs out,
and the extra epochs are not merely wasted — they overfit, so the model you finish with is
worse than one you already passed through.

Early stopping now defaults on with a patience of 20 epochs and **restores the best
weights**. That changes `epochs` from a prescription into a ceiling, which is the more
useful mental model: set it generously and let the run decide when it is done.

## 6. Hardware reality on the development machine

The GTX 1080 here runs at **240 MHz of a possible 1911** at 87 °C — about 12% clock, at
idle. Every timing in the README is therefore CPU, and examples are sized so that a run
finishes while you are still interested. This is a machine-specific fault, but it enforced a
useful discipline: if an example needs a healthy GPU to be bearable, it is a bad example.

## 7. Testing approach

`check.py`, 140 checks, most needing no ComfyUI.

The load-bearing one is per-layer round-tripping: trace → build → verify inferred shape
against a real forward pass → export source → `exec` it → load the trained weights in →
assert both models produce the same numbers. If the exported file ever stops being the model
that ran, that fails.

Error messages are tested as a feature: each assertion checks the message names its layer
and carries a non-empty hint. Two checks assert a network *fails* — a single Linear cannot
solve XOR, and it must not.

Example workflows are validated structurally — every node they reference exists, and every
saved widget value still fits the schema slot it lands in (§3.5) — rather than being
hand-written: they are built and run in the real editor, then serialised out. A hand-written
workflow JSON is a liability; a serialised one is a record of something that worked.

That is only half of it, though. The structural checks would not have noticed that the
Trainer's numbers were nonsense, because they were still numbers. What caught it was looking
at a screenshot. Every figure in the README is now taken from a run of the file it documents,
and the numbers quoted beside it come from that same run.

## 7b. Notes and groups in an example

`Note` and `MarkdownNote` are **frontend-only** node types — they are in litegraph's
registry and not in `/object_info`, so the backend never sees them and they cannot break a
prompt. `LGraphGroup` serialises as `{id, title, bounding, color, font_size, flags}` and is
added with `graph.add()` like a node. Both are therefore available to a workflow built
programmatically, which is what the diffusion example uses: six markdown notes explaining
each stage, and six coloured groups drawn round encoder, bottleneck, decoder, data, training
and sampling.

Worth doing deliberately. An example workflow that explains itself on the canvas is worth
more than the same graph plus a paragraph in the README, because the reader is looking at
the canvas.

One consequence for the screenshots: a note's text is a **DOM overlay**, exactly like the
image previews (§8), so it never reaches the canvas bitmap. A capture of the annotated
workflow is a picture of six empty boxes. The notes are therefore parked outside the frame
the README figure is cropped to — the figure shows the groups, and the README says the same
things the notes do.

## 8. Capturing the graphs

Screenshots of the canvas are taken in-page: fit the graph, resize the canvas backing store,
`toDataURL`, and POST the base64 to `/api/userdata/shots/<name>.b64`, which avoids moving
images through the conversation. Two things learned:

- **LiteGraph culls node text below about 0.6 zoom.** A first pass framed the whole U-Net at
  once and produced a picture of grey rectangles. Anything meant to be read needs scale ≥ 0.8,
  which for a wide graph means a canvas several thousand pixels across.
- **Node image previews never appear in a `toDataURL` capture.** They do appear in a real
  screenshot of the page, at any canvas size and any zoom, so they are composited over the
  LiteGraph canvas rather than drawn into it. The structural figures therefore show wires,
  widgets and shape badges but no previews; the one screenshot in the README that shows
  activations rendered on the nodes is a real screen capture.

## 9. Open

- No scheduler support (cosine, step decay). Would slot into `TrainConfig` cleanly.
- Nothing produces a *labelled* multi-input dataset, so a diffusion model still cannot be
  told which shape to draw. The machinery is there now (§4c); what is missing is a node that
  pairs a picture with its class.
- The diffusion example is the slowest thing in the pack at about eight minutes, and its
  samples are recognisably from the right world rather than recognisably shapes. A bigger
  model or a longer run fixes it; a bigger model or a longer run is not an example.
- No conditioning of any kind — a diffusion model that could be told *which* shape to draw
  needs a label on the wire, which is the same multi-input problem.
- Attention maps and saliency were scoped out of the inspect nodes and remain unbuilt.
