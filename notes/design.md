# Design — resolving torch against a node graph

The brief: build a ComfyUI nodepack where nodes are neural-network layers, exploiting the
fact that ComfyUI's slot typing already teaches people what may connect to what. Start from
how torch does it, then from how someone who doesn't code would think about it, and find the
interface in between that gives up neither nuance nor intuition.

---

## 1. How torch does it

Two distinct things, which torch deliberately blurs:

```python
lin = nn.Linear(64, 128)     # (a) a module: owns parameters, declares in/out features
y   = lin(x)                 # (b) a call: consumes a tensor, produces a tensor
```

A `Module` is a *thing with weights*. Calling it is an *event*. `nn.Sequential` is sugar for
"call these in order". The actual computation graph only exists transiently, inside autograd,
during a forward pass. Shape errors are runtime errors, discovered on the first batch.

Three properties of torch that a visual editor has to reckon with:

1. **A module is reusable.** Calling the same module twice shares weights. This is how
   Siamese networks and tied embeddings work.
2. **Shapes are implicit.** `nn.Linear(64, 128)` names its input width; `nn.Conv2d` names its
   channel count but not H/W; `nn.Flatten` names nothing at all. You cannot tell what a
   `Sequential` accepts without running it.
3. **The forward pass is arbitrary Python.** Residual adds, concatenation, multiple inputs,
   branching. `Sequential` covers maybe 60% of real architectures and 0% of interesting ones.

## 2. How a non-coder thinks about it

Ask someone who has seen a diagram of a neural network but never written one:

- A network is **boxes with arrows between them**. Data goes in the left, comes out the right.
- A layer is a **box you drop in**, and it has **knobs** ("how many neurons?").
- The thing on the wire is **the data, at that point in the network**.
- "What shape is it here?" is the question they ask constantly, and diagrams never answer.
- They do not distinguish "the layer" from "the layer's output". They point at a box and say
  "this is 128 wide" — meaning the activation, and the weight matrix, at once.
- Training is a **button**, not a loop. Its result is **a curve that goes down** and a picture
  of whether the thing works.

Note what they get *right*: they think of the data flowing, not of objects being constructed.
That intuition is worth more than it looks.

## 3. The interface in between

**The wire carries a tensor. The node is a layer. Dropping a layer on a wire is the same act
as constructing a module and calling it.**

That is not a compromise invented for this project — it is the Keras Functional API, and it is
the only mainstream NN interface that is already a DAG of typed values:

```python
x = Input(shape=(28, 28, 1))
h = Flatten()(x)
h = Dense(128, activation="relu")(h)
y = Dense(10)(h)
model = Model(inputs=x, outputs=y)
```

Read that vertically and it *is* a ComfyUI workflow. Every line is a node, every `h` is a wire.
Which means the answer to "what does the interface in between look like" is: **it already
exists, and ComfyUI is a better editor for it than Python is.**

So the mapping is:

| torch / Keras | neurodes |
|---|---|
| `Input(shape=…)` | **Input** node, takes a `SHAPE`, emits a `TENSOR` |
| `Dense(128)(h)` | **Linear** node: `TENSOR` in → `TENSOR` out, `units` widget |
| `h + skip` | **Add** node, two `TENSOR` in |
| `Model(inputs, outputs)` | **Build Model** node → `MODEL` |
| `model.compile(...)` / `.fit(...)` | **Train** node → `TRAINED`, `HISTORY` |

### What the wire actually carries

Not a real tensor — a **symbolic** one: a shape, a dtype, and a handle to the op that produced
it. Executing the ComfyUI graph *traces* the network rather than running it.

This is forced on us by the runtime (V3 `execute` is a cached, stateless classmethod, so real
`nn.Module`s built per-node would have chaotic parameter lifetimes), but it is also just
better:

- The graph executes in **milliseconds**, so the editor stays interactive.
- **Shape inference runs at design time.** Every node knows and can display its output shape
  before any data exists. That is exactly the question beginners keep asking and diagrams
  never answer.
- Errors land on **the node that caused them**, at edit time, with a message that can say
  `Linear expects the last dim to be 64, but got 128 — did you mean to Flatten first?`
- A trace is data, so it can be **compiled to readable PyTorch source**. The workflow is not a
  toy that mimics a network; it is a program that prints its own torch implementation.

That last point is the teaching payoff. The user drags boxes; the pack hands back the file a
practitioner would have written. Nothing is hidden, so nothing is lost when they graduate.

### Where the nuance is kept

The intuitive reading ("box = layer") loses three things from torch. Each is bought back
explicitly rather than papered over:

**Weight sharing.** Forking a wire duplicates *data*, not weights — the naive reading gets
this wrong. Every layer node has a `share` string widget: same non-empty tag + identical
config ⇒ one parameter set, used twice. Visible on the canvas, one concept to learn, and it
covers Siamese nets and tied embeddings.

**Symbolic dimensions.** Batch size and sequence length are unknown while designing. Shapes
are therefore lists of `int | name`: `B, 3, 224, 224` or `B, T, 512`. Named dims unify
wherever they meet and mismatches are reported by name, which is how people already reason
about shapes on a whiteboard.

**Repeated blocks.** A 12-layer transformer is not 12 nodes anyone wants to wire. Composite
nodes (`Transformer Block`, `Residual Block`, `MLP Block`) take a `repeat` count, so depth is
a knob. The primitives stay available for anyone who wants to build the block by hand — and
the emitted source shows the loop.

**More than one input.** A model can have several `Input` nodes, and a dataset can supply
several things — a pair to compare, a picture plus the timestep saying how noisy it is.
`DataBundle` keeps `x_train` as the first input and carries the rest in `side_train`, so
every single-input path is unchanged and the multi-input one is `model(*inputs)`. They are
fed in the order the `Input` nodes were traced, and a count mismatch is reported by name
before training starts.

### Training and inference are the same graph

A graph that fits a model and a graph that uses one differ by a single node. Everything
upstream of **Build Model** is the architecture; what comes after decides what happens to it:

| node | what it does with the model |
|---|---|
| **Train** | fits it, returns the fitted one |
| **Evaluate** | scores it on a split |
| **Forward (Images)** | one forward pass, pictures in and pictures out |
| **Predict (Images)** | the same for a classifier, with class names |
| **Sample (Diffusion)** | runs the reverse loop until a picture appears |
| **Capture Activations** | a forward pass that also keeps every intermediate |
| **Deep Dream** | a *backward* pass into the input, holding the weights still |

That is the design paying off. Because the model is a value on a wire rather than a mode the
graph is in, "train it" and "use it" are two nodes with the same input socket, and swapping
one for the other is a normal ComfyUI edit. It also means there is nowhere for a training-only
assumption to hide: `Forward (Images)` and `Train` are given the same object.

## 4. Why not the alternatives

**Eager execution** (each node builds a real module and runs a real batch). Rejected: node
output caching makes parameter identity unpredictable across runs, every edit costs a forward
pass, and there is no separation between "the architecture" and "a run of it".

**`nn.Sequential` only** (a chain node with a list of layers). Rejected: it throws away the
one thing ComfyUI is good at. Residual connections, two-input losses and multi-head outputs
are the interesting part of architecture, and a chain cannot express them.

**Config-object nodes** (each node emits a dict, one big builder assembles them). Rejected:
the wire stops meaning anything. `TENSOR` in / `TENSOR` out is what makes the type system
teach; `LAYER_CONFIG` in / `LAYER_CONFIG` out teaches nothing and would connect anything to
anything.

## 5. Consequences that shape the code

- **One registry, four consumers.** Each layer is declared once, with `infer` (shape math),
  `build` (`nn.Module`), `emit` (source line) and `params` (widget list). From that single
  table we generate the discrete nodes, the mega `Layer` node, the code exporter and the
  summary table. ~50 layers, no per-node boilerplate, and no way for the shape math to
  disagree with the module that gets built.
- **Core is pure Python.** `neurodes/core/` imports nothing from ComfyUI, so the whole thing
  is testable headless with `python check.py`. `neurodes/nodes/` is a thin adapter.
- **Shape errors are the product.** Every `infer` raises `ShapeError` with the layer name, the
  shape it got, the shape it wanted, and a suggested fix. This is the single highest-leverage
  surface in the pack and gets tested like it.
- **Two entry points for shape.** A text widget (`B, 3, 224, 224`) is primary — arbitrary rank,
  supports named dims, one widget. `Shape From Dims` uses `Autogrow` INT sockets for shapes
  computed upstream, since `Autogrow` forces widgets to sockets anyway (research.md §3).
- **Zero new dependencies.** torch + numpy + PIL only, all of which ComfyUI already requires.
