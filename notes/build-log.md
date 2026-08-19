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

## 4d. A language model, and the test that lied

Text needed three things: a dataset that pairs each window with itself shifted one
character, a loss that scores **every position** rather than one answer per example, and
causal masking. Only the first two were missing — `causal` was already a widget on Self
Attention and Transformer Block, which is what made this a day's work instead of a week's.

The loss change is four lines: a `[batch, time, vocabulary]` output against a
`[batch, time]` target folds both down so torch's cross entropy sees one row per position.
Everything else — early stopping, the charts, the compatibility checks — worked unchanged.
The one new check is worth having: a model that pools the time axis away now gets told
"each target is a sequence, so the model has to answer at every position" instead of a
tensor-size error naming nothing.

**The interesting part was the causal test.** With the mask off, a language model can read
the answer one step to the right, so it copies instead of predicting: 98% validation
accuracy, 0.07 loss, and it generates a wall of newlines. That is the single best
demonstration in the pack — a score four times better attached to a model that has learned
nothing.

The first version of the test scored **31% with the mask off** and looked like the leak did
not exist. The cause: I had left out **Learned Positions**. Attention with no positional
signal is permutation-invariant, so it *cannot express* "attend one to the right" and cannot
find the shortcut. The leak needs somewhere to read the position from.

That is a genuinely useful thing to have learned by getting it wrong, and it is why the
finished test builds the whole stack rather than the smallest thing that seemed sufficient.
A test that passes for the wrong reason and a test that fails for the wrong reason are the
same mistake.

## 4e. The corpus is the repo

`examples/sample_text.txt` is this project's own README and notes with the markdown taken
out — 55,000 characters of English about node graphs. No download, no licence question, and
the model learning to write about tensors and widgets is a better demonstration than the
same model learning to write about anything else.

It is small, and the model overfits it in single-digit epochs. Left as it is: the
overfitting is visible on the loss curve, early stopping deals with it, and "55,000
characters is a small corpus" is a fact worth meeting early. Doubling the dropout was tried
and bought exactly nothing (validation loss 2.4962 against 2.4944) for three times the wall
clock, which says the corpus is the ceiling rather than the regularisation.

Typographic characters were folded to ASCII first — em-dashes, ×, π, ⇒ — because each
appeared once or twice and cost an embedding row the model could never learn anything about.
97 distinct characters down to 87.

## 4f. Learning kernels with nothing to copy

The ask was a network that learns "novel" convolutional kernels, trained on the signal of
the image histogram changing a lot, and — correctly intuited in the same sentence — probably
needing some sort of guidance. Both halves turned out to be right, and the *way* the naive
version fails is worth more than the working version.

**The naive objective, implemented faithfully, learns nothing.** `histogram change`
maximises the L1 distance between the soft histogram of the response and the input's. On a
beach photograph it produces sixteen tiles of static: sparsity 0.619 where untrained kernels
score 0.621. Two separate faults, and separating them is the whole lesson:

- **Not blind to scale.** The score changes if you multiply the response by a constant, so
  the optimiser optimises the constant. Which direction it runs is data-dependent and I
  guessed wrong first: I wrote "it reaches for the largest number it can" into the module
  docstring, then measured **loudness 0.039** — a response 26× *smaller* than the input. On
  mean-centred patches the cheapest histogram maximally unlike a photograph's is a spike at
  zero, so the bank switches itself off. Turn `diversity` on and it bolts the other way to
  75×. Either way the volume knob is what gets trained. The fix is a scale-invariant
  statistic: `sparse response` is E|r| over the rms, and gain cancels exactly.
- **Silent about duplication.** Each kernel is scored alone, so all sixteen walk to the same
  minimum. This is not fixed by a better statistic — the good objective collapses too, to
  0.867 mean overlap. Worse, **the collapsed bank scores better**: 0.077 of sparsity gained
  against 0.069 for the diverse one. Sixteen copies of the best answer are sixteen good
  answers as far as the objective can see, the loss curve is genuinely lower, and nothing in
  the curve can ever reveal it. `diversity` supplies the missing sentence.

**The DC term needed isolating, not asserting.** Removing each patch's mean and forcing each
kernel to sum to zero do the same job, and I first measured them together and concluded the
kernel constraint did nothing. Isolated: neither on gives kurtosis 8.4, either one alone
gives 19.5, both gives 19.4. You want exactly one. Patch centring won because it is a
property of the dataset node and works for any architecture downstream.

**The reference I nearly shipped was the wrong one.** These statistics measure
non-Gaussianity, and photographs are extremely non-Gaussian before any filter touches them.
Quoting the textbook Gaussian numbers (0.798, kurtosis 3) would have presented the
*picture's* statistics as the model's achievement. Measured: the same random kernels score
0.799 on Gaussian noise exactly as theory says, **0.62 on a beach photo and 0.52 on a
screenshot of text** — so an untrained bank on the screenshot beats a well-trained bank on
the beach. Every report now re-initialises the model on the user's own data to measure the
floor, and the verdicts are phrased on the gap. This is the same trap as a size-matched null:
the floor has to come from the data, not from theory.

**Two rendering bugs, both of which made a working model look broken.**

- The response histogram drew trained and untrained curves on top of each other while every
  per-kernel statistic said they were nothing alike (kurtosis 14.2 vs 5.6). Cause: pooling
  kernels of different variances. A mixture of differently-scaled bells is heavy-tailed
  whatever the bells are. Standardising each kernel before pooling fixed it. A second pass
  was still illegible because the *Gaussian reference curve's* tail — about 1e-9 at five
  standard deviations, far below anything a finite sample shows — was setting the y-range
  and squashing the real difference into the top tenth of the plot.
- The filter bank rendered as flat grey. `signed` normalisation divides by the largest
  absolute value, and a sparse response map is by construction mostly small with a few huge
  spikes, so everything that was not a spike landed in a narrow band around mid grey. Added
  `signed, robust`, which scales by the 99.5th percentile and clips. Dividing by the maximum
  is exactly wrong for the thing this objective is built to produce.

**`Filter Bank` exists because `Capture Activations` resizes.** The natural way to show what
a learned kernel does is to slide it over the whole picture, and I assumed the existing
inspect nodes covered it. They do not: they convert the image to the model's declared input
size, so a 12×12 patch model got a 12×12 image and produced a 1×1 response. The new node
keeps full resolution, which also demonstrates the property that makes convolutions worth
having — the same weights run at any size.

## 4g. The seed did not mean what the node said it meant

Caught while building §4f's "set diversity to 0 and run again" demo, which quietly gave the
wrong answer: overlap 0.474 instead of 0.867, because the second run had *continued
training* the first run's kernels.

ComfyUI does not re-execute a node whose inputs have not changed. `Build Model`'s widgets do
not change when you edit the trainer, so it is never re-run, so the **same model object**
arrives at the trainer every time — carrying whatever weights the last run left in it.
Changing one number and pressing Run therefore compared a setting against itself plus more
training. Confirmed on plain `Train` with only the seed varied: 1.0740 → 1.0428, then
starting at 1.0426, then at 0.9591. And `Train`'s own description read *"Change the seed to
train again from scratch."* The node was documenting behaviour it did not have.

Fixed with `CompiledModel.reinitialise()` and a `reset_weights` flag defaulting on, appended
to the end of both trainers' widget lists so no saved workflow shifts. Two details:

- **Order matters, and got it wrong first.** Re-initialising before `model.to(device)` draws
  the first run's weights from the CPU generator and every later run's from the CUDA one —
  same seed, different weights, but *only on the first run*. Runs 2, 3 and 4 were
  byte-identical to each other, which is what gave it away. The move now happens first.
- Tied weights survive: a shared layer is one module used at two call sites and
  `modules()` yields each object once. There is a test.

`Evaluate` also had a latent `KeyError` on any unsupervised dataset — it indexed the loss
table directly rather than going through the constructor.

## 4h. A CNN taught one layer at a time

The single-layer bank of §4f can only ever show what a filter is. Depth is what makes a
convolutional net worth building, and the failure at the end of §4f turned out to be the way
in: **deep dream degenerates on a linear model**, because gradient ascent on a linear
function has no interior maximum. Add a ReLU and a second convolution and there is suddenly
something to find. The thing that did not work becomes the demonstration of why depth exists.

**One widget, and both halves of it are load-bearing.** `Discover Kernels` gained a `layer`
field. Naming a layer hands the optimiser *only* that layer's parameters, and moves the loss
to *that layer's* output. Doing one without the other is a different algorithm:

- freeze without moving the loss → end-to-end training with one layer unlocked, and the
  layers above dictate what this one learns;
- move the loss without freezing → the objective reaches down and rewrites the layers below
  to make its own job easier.

Freezing is done by handing the optimiser a subset of parameters, never by touching
`requires_grad`. §3.2 is why: the model object is shared across runs, and switching
gradients off leaves it permanently untrainable somewhere else. The one wrinkle is that
gradients still flow back through the frozen layers, so `model.zero_grad()` has to clear
every parameter rather than only the optimiser's, or `.grad` piles up down there for the
whole run. Never applied, never cleared.

Measured on a portrait, three layers of 16/32/48 kernels:

| layer | sparsity gained on its floor | overlap | draws on |
|---|---|---|---|
| conv2d_1 | 0.265 | 0.31 | pixels |
| conv2d_2 | 0.200 | 0.28 | 3.9 of 16 |
| conv2d_3 | 0.080 | 0.47 | 6.9 of 32 |

Two things worth having measured rather than assumed. The middle column is the answer to
"does a second layer learn anything, or just rename the first" — a participation ratio near
1.0 would mean each kernel reads a single feature below it, and 3.9 means real composition.
And the gains shrink while the overlap grows, so three layers is where this stops paying at
this size. That is in the example as a note; the alternative was implying it keeps going.

**The dream defaults were wrong for this and it hid the whole result.** The first pass came
back as diagonal hatching at every depth — layers 2 and 3 indistinguishable — which looked
like the ladder not existing. Two settings fixed it:

- **`objective="max"`, not `"mean"`.** Maximising the mean over a whole feature map asks
  "what makes this filter fire *everywhere*", and the answer is always the cheapest tileable
  texture. `max` asks what a single unit wants, and the answer is a feature on a blank field.
  This is the difference between the picture showing depth and not.
- **`smoothness` 0.4 and a *small* canvas.** 96 pixels, when layer 3's receptive field is
  60: on a large canvas the feature is a speck.

**Two bugs the work surfaced.** `reset_weights` was in `Discover Kernels`' schema but never
passed into the config, so the widget did nothing at all. And `loudness` was measured against
the *model's* input, which compounds with depth — layer 3 read as 134x and tripped the
runaway-gain warning when nothing was wrong. It is now measured against whatever feeds the
layer (2.6 / 4.5 / 6.8), and the warning additionally requires that nothing was learned,
since "paid in volume rather than structure" is only a fault when the structure is missing.

**And the widget-drift bug, for the third time.** `layer` was first added at the *front* of
the input list, which shifted every saved value in example 14. The alignment check from §3.5
caught it on the next run, which is the first time that check has paid for itself on a
mistake made after it was written. New widgets go at the end. Always.

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

`check.py`, 170 checks, most needing no ComfyUI.

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

## 8a. Generating a workflow file, and the three ways it went wrong

Example 14 is emitted by a script that reads the live schemas, rather than hand-written,
specifically to avoid the widget drift of §3.5. That worked — the widget values were right
first time — and then the *sockets* were wrong in three separate ways, none of which any
existing check caught, and all of which the editor showed as a red border on the node.

- **The autogrow container is not a socket.** `Build Model` declares one input of type
  `COMFY_AUTOGROW_V3`; the frontend spawns numbered slots `outputs.output0`,
  `outputs.output1` from it and wires go on those. Saving the container itself gives a link
  whose type is `COMFY_AUTOGROW_V3` where a `NEURO_TENSOR` belongs.
- **`control_after_generate` has no socket at all.** It appends a combo to `widgets_values`
  — hence `[1030524549, 'randomize', True]` for two widgets — but never adds an input. The
  generator emitted a `seed control` input the schema does not declare.
- **A MultiType socket keeps its whole type list.** An Input's shape is
  `STRING,NEURO_SHAPE`, because it takes typed text *or* a wire from a dataset. The
  generator used `get_io_type().split(",")[0]` to decide which widget to draw — correct for
  that — and reused the truncated string as the socket type, so the editor rejected the wire
  already attached to it.

The lesson is the same one as §3.5 and it did not transfer: **a workflow JSON has two
independent surfaces that can drift from the schema, and checking one proves nothing about
the other.** There is now a link check next to the widget check. It walks every wire in
every example, asserts the target input exists on the node's schema, that no raw autogrow
container was saved as a socket, that the link id is in the link table, and that the type
leaving the source is one the destination accepts — MultiType included. 241 wires, and it
found all three on the first run.

## 8b. A synthetic photograph for the tests

The discovery checks need an image with natural-image statistics and no download. The
obvious choice is noise shaped to a 1/f spectrum, and it is wrong: shaping the spectrum
leaves the phases random, and phase alignment is what makes a picture a picture. Random-phase
1/f noise is still Gaussian, contains no lines or boundaries, and untrained kernels score
0.797 on it against a Gaussian's 0.798. Built on that image the tests were actively
misleading — the naive objective passed and the working one failed, because there was
nothing to find and finding nothing was the correct answer.

Replaced with a dead-leaves model: opaque discs of power-law-distributed radius, random
position and random shade, painted one over another. Occlusion boundaries at every scale and
orientation. Untrained kernels now score 0.62 and a trained bank reaches 0.33.

One detail the first version got wrong: discs are flat inside, so a patch cut from within
one becomes exactly zero once its mean is removed, and two unrelated patches then compare
byte-identical. The train/validation-leakage test accused itself. A trace of grain fixes it.

## 4i. Five state-of-the-art shapes, each shipped with the control that tests it

Examples 16 to 20. Every claim below is measured on a 6,000-image CIFAR-10 slice, not
quoted from a paper. Three of the five landed somewhere other than the folklore, and in each
case the interesting part is *why*.

**Residual networks — the gap is a function of depth, and shallow demonstrations prove the
wrong thing.**

| convolutions | residual | plain | gap |
|---|---|---|---|
| 14 | 0.6150 | 0.5945 | +0.021 |
| 20 | 0.5945 | 0.5680 | +0.027 |
| 26 | 0.5950 | 0.4760 | +0.119 |
| 32 | 0.5920 | 0.4660 | +0.126 |

Measured at 14 layers first and nearly wrote "the skip is marginal" — true, useless, and the
opposite of the point. Read down the columns instead of across: the residual net is flat
(0.615, 0.595, 0.595, 0.592) and the plain one falls off a cliff between 20 and 26. The
example uses 26 because it is the shallowest depth where the effect is unmistakable.

`Residual Block` gained a `skip` flag for this, appended so nothing shifts. Off, the
addition goes and **the 1×1 projection is not built at all** — which is why the plain stack
has *fewer* weights, 366,938 against 369,690. The better model is bigger because it is
better. Both counts reproduce a hand-built version exactly, so the control is provably the
same code path minus one wire.

**Vision transformer — it loses, and the way it loses is the finding.** ViT 0.4390 against a
plain CNN's 0.5415, with 5.7× the parameters. And a ViT with a fifth of the parameters
scores 0.4415 — *identical*. It is not short of capacity, it is short of data. A convolution
gets locality and translation-invariance for free; attention has to learn them from
examples, and 6,000 is not enough examples.

**Depthwise separable — 539,178 weights to 67,914, for 0.6095 → 0.5925.** The 7.9× is
arithmetic and cannot vary. The accuracy line is two seeds and is written as "it did not
collapse" rather than "it cost nothing".

**Inception — 29,578 weights against 82,698, and slightly better.** Variety is cheaper than
width: the control has to be wide to be expressive and width in a 3×3 costs `in × out × 9`,
while three of inception's four branches are 1×1 or preceded by one.

**Squeeze-and-excitation — nothing.** 0.6085 with, 0.6150 without, for 5,656 extra
parameters. Inside the spread and the wrong sign. It stays in example 16 as a documented
null result, because a workflow that quietly dropped the measurement would imply otherwise.

**Mixture of experts — a claim that died, and a better one underneath it.** A 3-seed run put
the mixture ahead of a matched dense control by +0.075. At 8 seeds on two devices the gap is
+0.059 on CPU and **−0.021 on CUDA**, against a combined spread of 0.065–0.082: no
difference in the means, and the sign flips with the device.

What is real is the *spread*. The 51-parameter dense control ranges from **0.457 to 0.741**
on the seed alone — bimodal, sometimes solving it and sometimes collapsing. The mixture's
spread is 0.008. At a matched budget routing does not buy a better answer, it buys the same
answer every time. This is §6's size-matched-noise lesson arriving from a new direction:
**three seeds is not a measurement of a model whose standard deviation is 0.08.**

Two further things the mixture example says out loud. Purity inside each expert's region is
0.37–0.51 against a 0.33 floor, because a linear gate cuts the plane into wedges and spiral
arms cross all of them — it partitions space, not meaning. And on `blobs`, two of four
experts received zero points: expert collapse, live, with no load-balancing term to stop it.
The gate is soft, so every expert runs every time and **there is no compute saving at all** —
the actual reason frontier models use mixtures needs top-k routing, which this pack cannot
express.

`Decision Boundary` gained a `layer` widget to draw the routing: name a layer and the
background becomes that layer's argmax, muted so region colours cannot be mistaken for the
class colours sitting on them. A gate has already been through a softmax, so
`_as_distribution` passes a distribution through untouched — a second softmax would leave
the argmax alone but flatten the peak, and the peak is what the confidence fade draws.

## 8c. The autogrow prefix, and a check that was still not enough

§8a added a link check and called the socket problem solved. It was not.

The numbered slots an autogrow input spawns carry a **per-node prefix**. `Build Model` grows
`outputs.output0`; every variadic *layer* — `Concat`, `Add`, `Multiply` — grows
`tensors.tensor0`. The generator hardcoded `output` for both. Build Model was right by
accident and every `Add`, `Multiply` and `Concat` was wrong.

The check from §8a validated only the container root, `tensors`, which exists. So
`tensors.output0` passed all 189 checks while being unloadable.

What ComfyUI does with it is the dangerous part: it fails validation, **drops the node and
its entire downstream, and still reports the run a success**. Example 20 "succeeded" in 88
seconds with only its control branch having run. Example 19 shipped the same way and was
reported here as working, because only the tail of its log was read and the
`produced nothing` line prints above the outputs.

Three fixes. The generator now reads `template.prefix` and `template.min` off the schema
instead of assuming. The check now validates the suffix against the prefix, the index against
`max`, and the wired count against `min`. And the check was verified by *reintroducing the
bug* — the first attempt at that test was itself wrong, producing `outputs.tensor0` which
tripped the older root assertion, so it would have passed without ever exercising the new
one.

A related trap found the same day: **the server ignores inputs it does not know.** ComfyUI
loads custom nodes once, at startup, so a running server with stale code silently discarded
`skip` and `layer` rather than erroring — every `Residual Block` in example 16 trained *with*
its skip and the comparison was void, while reporting success. Check `/object_info/<node>`
for a new widget before trusting a run that depends on it.

## 9b. One bit per press

`core/preference.py` learns a parameter mapping from nothing but a human pressing like or
dislike. Simulated against a hidden taste before any node was written, which was worth doing
because the defaults it shipped with were wrong in the most embarrassing direction.

**Sixty gradient steps per press measured worse than never training** — error up 0.077,
memorising a handful of noisy proposals rather than learning. One step per press turns that
into −0.186. The default is now 1, with the number in the docstring so nobody helpfully turns
it back up.

**A picky human teaches it less than a generous one.** Tolerance 0.28 improved error by
0.138; tightening to 0.10 made it 0.022 *worse*, because 1% of proposals were ever liked and
advantage-weighted regression has nothing to pull towards. Withholding approval starves it.

**It works for a few knobs.** Four outputs 0.341 → 0.101; eight outputs 0.318 → 0.212.

The reward model earns its place — 4 outputs 0.101 → 0.076, 8 outputs 0.212 → 0.181, same
votes — and then stops. At eight outputs *with* two inputs, a ten-dimensional pair space
covered by 120 votes, turning the critic on made it worse: −0.083 without, −0.051 with.
Reward hacking, reproduced small enough to watch. `report()` prints the critic's confidence
next to how many votes it was fitted on, because from the inside the critic always thinks it
is going well.

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
