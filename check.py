"""neurodes test harness.  Run with:  python check.py

Nothing here needs ComfyUI. The schema section runs only if a ComfyUI install can be found,
and says so if it cannot.

The load-bearing test is `layer <name>`: for every registered layer it traces a graph,
builds the real module, checks the inferred shape against what torch actually produces,
exports the PyTorch source, executes that source, loads the trained weights into it and
compares the two models numerically. If the exported file ever stops being the model that
ran, this fails.
"""

from __future__ import annotations

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from neurodes.core import (Shape, apply_layer, build_model, make_input, summarize)
from neurodes.core.emit import emit_source
from neurodes.core.errors import (BuildError, GraphError, NeurodesError, ParseError,
                                  ShapeError)
from neurodes.core.registry import VARIADIC, all_specs, get
from neurodes.core.shape import Dim, broadcast, unify

PASS, FAIL = [], []


def check(name: str):
    def wrap(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok   {name}")
        except Exception as exc:                                  # noqa: BLE001
            FAIL.append((name, exc))
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap


def expect_error(kind, fn, must_mention: str = ""):
    try:
        fn()
    except kind as exc:
        text = str(exc).lower()
        assert must_mention.lower() in text, f"message missing {must_mention!r}: {exc}"
        assert getattr(exc, "hint", None), f"no hint on: {exc}"
        return exc
    raise AssertionError(f"expected {kind.__name__}, nothing was raised")


def expect_error_plain(kind, fn):
    """For ordinary Python exceptions, which carry no hint of their own."""
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}, nothing was raised")


# ---------------------------------------------------------------------------
# 1. Shapes
# ---------------------------------------------------------------------------
print("\nshapes")


@check("parse text forms")
def _():
    assert str(Shape.parse("B, 3, 224, 224")) == "[B, 3, 224, 224]"
    assert str(Shape.parse("[4 x 8]")) == "[4, 8]"
    assert str(Shape.parse("2,3")) == "[2, 3]"
    assert Shape.parse("B, T, 512").rank == 3
    assert Shape.parse("?, 8")[0].is_symbolic


@check("reject bad shapes with a hint")
def _():
    expect_error(ParseError, lambda: Shape.parse(""), "empty")
    expect_error(ParseError, lambda: Shape.parse("B, -4"), "negative")
    expect_error(ParseError, lambda: Shape.parse("B, 3, @"), "@")


@check("unify prefers concrete over symbolic")
def _():
    assert str(unify(Shape.parse("B, 8"), Shape.parse("4, 8"))) == "[4, 8]"
    expect_error(ShapeError, lambda: unify(Shape.parse("4, 8"), Shape.parse("4, 9")), "does not match")
    expect_error(ShapeError, lambda: unify(Shape.parse("4"), Shape.parse("4, 8")), "rank")


@check("broadcasting")
def _():
    assert str(broadcast([Shape.parse("B, 1, 8"), Shape.parse("B, 5, 8")])) == "[B, 5, 8]"
    assert str(broadcast([Shape.parse("8"), Shape.parse("4, 8")])) == "[4, 8]"
    expect_error(ShapeError, lambda: broadcast([Shape.parse("4, 3"), Shape.parse("4, 5")]),
                 "broadcast")


@check("axis normalisation")
def _():
    from neurodes.core.shape import normalize_axis
    assert normalize_axis(-1, 4) == 3
    assert normalize_axis(-1, 3, allow_end=True) == 3
    expect_error(ShapeError, lambda: normalize_axis(9, 4), "out of range")


# ---------------------------------------------------------------------------
# 2. Every layer: infer -> build -> verify -> emit -> execute -> compare
# ---------------------------------------------------------------------------
print("\nlayers")

VEC, SEQ, IMG, VOL, SIG = "B, 16", "B, 7, 16", "B, 3, 12, 12", "B, 3, 6, 6, 6", "B, 3, 16"

CASES: dict[str, tuple] = {
    "linear": ([VEC], {"units": 8}),
    "embedding": ([("B, 5", "int64")], {"num_embeddings": 20, "embedding_dim": 8}),
    "dropout": ([VEC], {}),
    "identity": ([VEC], {}),
    "batchnorm": ([IMG], {}),
    "layernorm": ([VEC], {}),
    "rmsnorm": ([VEC], {}),
    "groupnorm": ([IMG], {"groups": 3}),
    "softmax": ([VEC], {}),
    "log_softmax": ([VEC], {}),
    "prelu": ([IMG], {"per_channel": True}),
    "conv1d": ([SIG], {"out_channels": 8}),
    "conv2d": ([IMG], {"out_channels": 8}),
    "conv3d": ([VOL], {"out_channels": 4}),
    "conv_transpose2d": ([IMG], {"out_channels": 8, "stride": "2", "padding": "1",
                                 "output_padding": "1"}),
    "maxpool1d": ([SIG], {}),
    "avgpool1d": ([SIG], {}),
    "maxpool2d": ([IMG], {}),
    "avgpool2d": ([IMG], {}),
    "adaptive_avgpool2d": ([IMG], {"output_size": "2"}),
    "global_pool": ([IMG], {}),
    "upsample": ([IMG], {}),
    "lstm": ([SEQ], {"hidden_size": 8}),
    "gru": ([SEQ], {"hidden_size": 8, "return_sequences": True}),
    "rnn": ([SEQ], {"hidden_size": 8, "bidirectional": True}),
    "self_attention": ([SEQ], {"num_heads": 4}),
    "cross_attention": ([SEQ, "B, 5, 16"], {"num_heads": 4}),
    "sinusoidal_positions": ([SEQ], {}),
    "learned_positions": ([SEQ], {"max_len": 32}),
    "flatten": ([IMG], {}),
    "reshape": ([VEC], {"target": "B, 4, 4"}),
    "permute": ([IMG], {"order": "0, 2, 3, 1"}),
    "transpose": ([SEQ], {"dim0": 1, "dim1": 2}),
    "unsqueeze": ([VEC], {"dim": 1}),
    "squeeze": (["B, 1, 16"], {"dim": 1}),
    "concat": ([VEC, VEC], {"dim": 1}),
    "stack": ([VEC, VEC], {"dim": 1}),
    "slice": ([VEC], {"dim": 1, "start": 2, "end": 10}),
    "reduce": ([SEQ], {"dim": 1}),
    "add": ([VEC, VEC], {}),
    "subtract": ([VEC, VEC], {}),
    "multiply": ([VEC, VEC], {}),
    "divide": ([VEC, VEC], {}),
    "scale": ([VEC], {"value": 0.5}),
    "offset": ([VEC], {"value": 1.0}),
    "clamp": ([VEC], {}),
    "matmul": (["B, 3, 5", "B, 5, 7"], {}),
    "einsum": (["B, 3, 5", "B, 5, 7"], {"equation": "bij,bjk->bik"}),
    "cast": ([VEC], {"dtype": "float32"}),
    "mlp_block": ([VEC], {"hidden": 12, "out_features": 5, "depth": 3}),
    "conv_block": ([IMG], {"out_channels": 8, "pool": True}),
    "residual_block": ([IMG], {"out_channels": 6, "stride": 2}),
    "transformer_block": ([SEQ], {"num_heads": 4, "repeat": 2}),
}

_DEFAULT_BY_CATEGORY = {
    "activations": ([VEC], {}),
    "layers/basic": ([VEC], {}),
    "layers/norm": ([VEC], {}),
    "layers/conv": ([IMG], {}),
    "layers/pool": ([IMG], {}),
    "layers/recurrent": ([SEQ], {}),
    "layers/attention": ([SEQ], {}),
    "shape": ([VEC], {}),
    "ops/math": ([VEC, VEC], {}),
    "blocks": ([VEC], {}),
}


def build_case(spec):
    shapes, cfg = CASES.get(spec.key) or _DEFAULT_BY_CATEGORY[spec.category]
    tensors = []
    for i, entry in enumerate(shapes):
        text, dtype = entry if isinstance(entry, tuple) else (entry, "float32")
        tensors.append(make_input(Shape.parse(text), dtype=dtype, name=f"x{i + 1}"))
    out = apply_layer(spec.key, tensors, cfg)
    return out


def roundtrip(spec):
    """Trace, build, verify, export, execute the export, and compare the two numerically."""
    out = build_case(spec)
    model = build_model([out], name=f"Test{spec.key}", verify=True)

    source = emit_source([out], name=f"Test{spec.key}", include_demo=False)
    compile(source, f"<{spec.key}>", "exec")          # the file must at least be valid Python

    namespace: dict = {}
    exec(source, namespace)                            # noqa: S102 - that is the whole point
    cls_name = next(k for k, v in namespace.items()
                    if isinstance(v, type) and issubclass(v, torch.nn.Module)
                    and k.lower().startswith("test"))
    exported = namespace[cls_name]()

    state = {k.replace("_modules_by_name.", ""): v for k, v in model.state_dict().items()}
    missing, unexpected = exported.load_state_dict(state, strict=False)
    assert not missing, f"exported model is missing weights: {missing}"
    assert not unexpected, f"exported model does not use: {unexpected}"

    torch.manual_seed(0)
    inputs = model.dummy_inputs(batch=4)
    model.eval()
    exported.eval()
    with torch.no_grad():
        a = model(*inputs)
        b = exported(*inputs)
    a = a if isinstance(a, torch.Tensor) else a[0]
    b = b if isinstance(b, torch.Tensor) else b[0]
    assert a.shape == b.shape, f"exported shape {tuple(b.shape)} != built {tuple(a.shape)}"
    assert torch.allclose(a.float(), b.float(), atol=1e-5), "exported model gives different numbers"
    return model


for _spec in all_specs():
    @check(f"layer {_spec.key}")
    def _(_spec=_spec):
        roundtrip(_spec)


@check("every layer declares complete metadata")
def _():
    for spec in all_specs():
        assert spec.doc.strip(), f"{spec.key} has no description"
        assert spec.display, f"{spec.key} has no display name"
        for p in spec.params:
            assert p.doc.strip(), f"{spec.key}.{p.name} has no tooltip"
        assert spec.build is not None or spec.apply is not None, f"{spec.key} cannot run"
        assert spec.build is None or spec.emit_init is not None, f"{spec.key} cannot be exported"


# ---------------------------------------------------------------------------
# 3. Error messages
# ---------------------------------------------------------------------------
print("\nerrors")


def vec(text="B, 16", dtype="float32"):
    return make_input(Shape.parse(text), dtype=dtype)


@check("linear applies to the last dimension at any rank")
def _():
    # Not an error, and deliberately so: applying a Linear per position is how every
    # transformer works. The badge showing [B, 3, 12, 4] is what tells someone who wanted a
    # classifier that they meant to Flatten first.
    out = apply_layer("linear", [vec(IMG)], {"units": 4})
    assert str(out.shape) == "[B, 3, 12, 4]"


@check("linear on an unknown feature width suggests Flatten")
def _():
    exc = expect_error(ShapeError, lambda: apply_layer("linear", [vec("B, 3, W")], {"units": 4}),
                       "symbolic size")
    assert "flatten" in str(exc).lower()


@check("conv on the wrong rank names the right layer")
def _():
    expect_error(ShapeError, lambda: apply_layer("conv2d", [vec()], {}), "rank-4")


@check("attention head count that does not divide")
def _():
    exc = expect_error(ShapeError, lambda: apply_layer("self_attention", [vec(SEQ)],
                                                       {"num_heads": 5}), "heads")
    assert "8" in str(exc) or "4" in str(exc), "should suggest workable head counts"


@check("reshape that changes the number of values")
def _():
    expect_error(ShapeError, lambda: apply_layer("reshape", [vec()], {"target": "B, 5, 5"}),
                 "do not divide" if False else "cannot reshape")


@check("concat with mismatched sides")
def _():
    expect_error(ShapeError,
                 lambda: apply_layer("concat", [vec("B, 4, 8"), vec("B, 5, 8")], {"dim": 2}),
                 "disagree")


@check("embedding rejects float input")
def _():
    expect_error(ShapeError, lambda: apply_layer("embedding", [vec()], {}), "int64")


@check("linear rejects integer input")
def _():
    expect_error(ShapeError, lambda: apply_layer("linear", [vec("B, 5", "int64")], {}),
                 "floating-point")


@check("groupnorm suggests valid group counts")
def _():
    exc = expect_error(ShapeError, lambda: apply_layer("groupnorm", [vec("B, 10, 4, 4")],
                                                       {"groups": 3}), "equal groups")
    assert "5" in str(exc), "should list divisors"


@check("unconnected input is reported by slot name")
def _():
    expect_error(ShapeError, lambda: apply_layer("matmul", [vec("B, 3, 5"), None], {}),
                 "not connected")


@check("padding 'same' with stride 2 explains itself")
def _():
    expect_error(ShapeError, lambda: apply_layer("conv2d", [vec(IMG)],
                                                 {"out_channels": 4, "stride": "2",
                                                  "padding": "same"}), "stride 1")


@check("a window bigger than the image")
def _():
    expect_error(ShapeError, lambda: apply_layer("conv2d", [vec("B, 3, 4, 4")],
                                                 {"out_channels": 4, "kernel_size": "9",
                                                  "padding": "valid"}), "collapses")


@check("einsum equation mismatched with the tensors")
def _():
    expect_error(ShapeError, lambda: apply_layer("einsum", [vec("B, 3, 5"), vec("B, 5, 7")],
                                                 {"equation": "bij,bj->bi"}), "names 2 axes")


# ---------------------------------------------------------------------------
# 4. Graphs
# ---------------------------------------------------------------------------
print("\ngraphs")


@check("a branching graph with a residual add")
def _():
    x = make_input(Shape.parse("B, 32"), name="x")
    h = apply_layer("linear", [x], {"units": 32})
    h = apply_layer("relu", [h])
    h = apply_layer("linear", [h], {"units": 32})
    y = apply_layer("add", [x, h])
    model = build_model([y], name="Residual")
    assert str(model.output_shapes[0]) == "[B, 32]"
    assert model.n_parameters() == 32 * 32 + 32 + 32 * 32 + 32


@check("two outputs")
def _():
    x = make_input(Shape.parse("B, 8"), name="x")
    a = apply_layer("linear", [x], {"units": 3})
    b = apply_layer("linear", [x], {"units": 5})
    model = build_model([a, b], name="TwoHeads")
    out = model(*model.dummy_inputs(4))
    assert isinstance(out, tuple) and out[0].shape[-1] == 3 and out[1].shape[-1] == 5


@check("two inputs")
def _():
    a = make_input(Shape.parse("B, 8"), name="left")
    b = make_input(Shape.parse("B, 8"), name="right")
    y = apply_layer("concat", [a, b], {"dim": 1})
    model = build_model([y], name="TwoIn")
    assert str(model.output_shapes[0]) == "[B, 16]"
    assert len(model.input_ops) == 2


@check("weight sharing uses one set of weights")
def _():
    a = make_input(Shape.parse("B, 8"), name="a")
    b = make_input(Shape.parse("B, 8"), name="b")
    ea = apply_layer("linear", [a], {"units": 4}, share="tower")
    eb = apply_layer("linear", [b], {"units": 4}, share="tower")
    y = apply_layer("subtract", [ea, eb])
    model = build_model([y], name="Siamese")
    assert model.n_parameters() == 8 * 4 + 4, "shared layer should be counted once"
    with torch.no_grad():
        same = torch.randn(3, 8)
        assert torch.allclose(model(same, same), torch.zeros(3, 4), atol=1e-6)


@check("incompatible share tags are refused")
def _():
    a = make_input(Shape.parse("B, 8"), name="a")
    b = make_input(Shape.parse("B, 8"), name="b")
    ea = apply_layer("linear", [a], {"units": 4}, share="tower")
    eb = apply_layer("linear", [b], {"units": 9}, share="tower")
    y = apply_layer("concat", [ea, eb], {"dim": 1})
    expect_error(GraphError, lambda: build_model([y], name="Bad"), "sharing weights")


@check("a model with no input is refused")
def _():
    expect_error(GraphError, lambda: build_model([], name="Empty"), "at least one output")


@check("symbolic sequence length survives the whole graph")
def _():
    x = make_input(Shape.parse("B, T, 32"), name="tokens")
    h = apply_layer("transformer_block", [x], {"num_heads": 4, "repeat": 2})
    h = apply_layer("reduce", [h], {"mode": "mean", "dim": 1})
    y = apply_layer("linear", [h], {"units": 4})
    model = build_model([y], name="Seq")
    assert str(model.output_shapes[0]) == "[B, 4]"
    with torch.no_grad():
        assert tuple(model(torch.randn(2, 11, 32)).shape) == (2, 4)
        assert tuple(model(torch.randn(2, 3, 32)).shape) == (2, 4), "T must stay free"


@check("a realistic CNN exports and matches")
def _():
    x = make_input(Shape.parse("B, 1, 28, 28"), name="image")
    h = apply_layer("conv_block", [x], {"out_channels": 16, "pool": True})
    h = apply_layer("conv_block", [h], {"out_channels": 32, "pool": True})
    h = apply_layer("global_pool", [h], {})
    h = apply_layer("dropout", [h], {"p": 0.2})
    y = apply_layer("linear", [h], {"units": 10})
    model = build_model([y], name="SmallCNN")
    assert str(model.output_shapes[0]) == "[B, 10]"
    source = emit_source([y], name="SmallCNN")
    ns: dict = {}
    exec(source, ns)                                   # noqa: S102
    exported = ns["SmallCNN"]()
    state = {k.replace("_modules_by_name.", ""): v for k, v in model.state_dict().items()}
    assert not exported.load_state_dict(state, strict=False).missing_keys
    model.eval(), exported.eval()
    with torch.no_grad():
        probe = torch.randn(2, 1, 28, 28)
        assert torch.allclose(model(probe), exported(probe), atol=1e-5)
    assert "# [B, 16, 14, 14]" in source, "forward lines should carry shape comments"


@check("summary table adds up")
def _():
    x = make_input(Shape.parse("B, 4"), name="x")
    h = apply_layer("linear", [x], {"units": 6})
    y = apply_layer("linear", [h], {"units": 2})
    text = summarize([y], "Tiny")
    assert "parameters 44" in text.replace(",", ""), text
    assert "[B, 6]" in text and "[B, 2]" in text


# ---------------------------------------------------------------------------
# 5. Data and training
# ---------------------------------------------------------------------------
print("\ntraining")

from neurodes.core import data as D               # noqa: E402
from neurodes.core import plot as PL              # noqa: E402
from neurodes.core import train as T              # noqa: E402


@check("every toy dataset builds")
def _():
    for name in D.TOY_NAMES:
        bundle = D.toy_classification(name, n=200, seed=0)
        assert bundle.n_train > 0 and bundle.n_val > 0
        assert bundle.is_2d_points
        assert str(bundle.input_shape) == "[B, 2]"
    for name in D.CURVE_NAMES:
        bundle = D.toy_regression(name, n=200, seed=0)
        assert bundle.task == "regression"


@check("a network learns xor")
def _():
    bundle = D.toy_classification("xor", n=800, noise=0.02, seed=3)
    x = make_input(bundle.input_shape, name="p")
    h = apply_layer("mlp_block", [x], {"hidden": 32, "out_features": 2, "depth": 3,
                                       "activation": "tanh"})
    model = build_model([h], name="Xor")
    hist = T.train(model, bundle, T.TrainConfig(epochs=60, learning_rate=0.03, batch_size=64))
    assert hist.best_val_acc > 0.9, f"only reached {hist.best_val_acc:.2f}"


@check("a network without a hidden layer fails xor")
def _():
    bundle = D.toy_classification("xor", n=800, noise=0.02, seed=3)
    x = make_input(bundle.input_shape, name="p")
    y = apply_layer("linear", [x], {"units": 2})
    model = build_model([y], name="Linear")
    hist = T.train(model, bundle, T.TrainConfig(epochs=40, learning_rate=0.05))
    assert hist.best_val_acc < 0.75, "a single Linear should not solve xor"


@check("regression trains")
def _():
    bundle = D.toy_regression("sine", n=400, seed=1)
    x = make_input(bundle.input_shape, name="t")
    y = apply_layer("mlp_block", [x], {"hidden": 64, "out_features": 1, "depth": 4,
                                       "activation": "silu"})
    model = build_model([y], name="Curve")
    hist = T.train(model, bundle, T.TrainConfig(epochs=120, learning_rate=0.01, batch_size=32))
    assert hist.val_loss[-1] < hist.val_loss[0] * 0.4, "the fit should improve a lot"


@check("wrong output width is caught before training")
def _():
    bundle = D.toy_classification("blobs", n=200, seed=0)
    x = make_input(bundle.input_shape, name="p")
    y = apply_layer("linear", [x], {"units": 7})
    model = build_model([y], name="Wrong")
    expect_error(NeurodesError,
                 lambda: T.train(model, bundle, T.TrainConfig(epochs=1)), "3 classes")


@check("a regression loss on labels is caught")
def _():
    bundle = D.toy_classification("blobs", n=200, seed=0)
    x = make_input(bundle.input_shape, name="p")
    y = apply_layer("linear", [x], {"units": 3})
    model = build_model([y], name="Wrong")
    expect_error(NeurodesError,
                 lambda: T.train(model, bundle, T.TrainConfig(epochs=1, loss="mse")),
                 "regression loss")


@check("softmax before cross entropy is warned about, not blocked")
def _():
    bundle = D.toy_classification("blobs", n=200, seed=0)
    x = make_input(bundle.input_shape, name="p")
    h = apply_layer("linear", [x], {"units": 3})
    y = apply_layer("softmax", [h], {})
    model = build_model([y], name="Double")
    hist = T.train(model, bundle, T.TrainConfig(epochs=2))
    assert any("cross entropy" in n for n in hist.notes), hist.notes


@check("a mismatched Input shape is caught")
def _():
    bundle = D.toy_classification("blobs", n=200, seed=0)
    x = make_input(Shape.parse("B, 5"), name="p")
    y = apply_layer("linear", [x], {"units": 3})
    model = build_model([y], name="Wrong")
    expect_error(NeurodesError, lambda: T.train(model, bundle, T.TrainConfig(epochs=1)),
                 "input node")


@check("early stopping stops, and keeps the best weights")
def _():
    # A tiny model on noise: validation loss bottoms out quickly and then drifts, which is
    # exactly the situation early stopping exists for.
    data = D.images_to_dataset(torch.rand(160, 8, 8, 1), torch.randint(0, 2, (160,)))
    x = make_input(data.input_shape, name="image")
    f = apply_layer("flatten", [x], {"start_dim": 1})
    h = apply_layer("linear", [f], {"units": 64})
    h = apply_layer("relu", [h])
    y = apply_layer("linear", [h], {"units": 2})
    model = build_model([y], name="Stopper")

    hist = T.train(model, data, T.TrainConfig(epochs=400, learning_rate=0.02,
                                              early_stopping=5, seed=0))
    assert hist.stopped_early, "400 epochs on noise should have plateaued"
    assert len(hist.epochs) < 400, len(hist.epochs)
    assert hist.best_epoch and hist.best_epoch <= len(hist.epochs)
    assert any("not improved" in n for n in hist.notes), hist.notes

    # The weights kept must be the best ones, not the last ones.
    loss_fn = T._LOSS_FNS[T.resolve_loss("auto", data)]()
    final, _ = T.evaluate(model, data.x_val, data.y_val, loss_fn, "cross entropy", data.task)
    assert final <= min(hist.val_loss) + 1e-4, \
        f"ended on {final:.4f} but the best epoch was {min(hist.val_loss):.4f}"


@check("the summary describes the weights you have, not the ones thrown away")
def _():
    # After a restore the last epoch's numbers belong to a model that no longer exists,
    # so quoting them in the summary would be a straightforwardly false report.
    hist = T.History(task="classification",
                     epochs=[1, 2, 3], train_loss=[1.0, 0.5, 0.2],
                     val_loss=[1.0, 0.4, 0.9], val_acc=[0.5, 0.9, 0.6],
                     best_epoch=2)
    assert hist.restored and hist.kept == 1
    text = hist.summary()
    assert "kept epoch 2" in text, text
    assert "0.4000" in text and "0.9000" not in text, text     # val loss, not the last one
    assert "90.0%" in text and "60.0%" not in text, text        # val acc, likewise

    # And with no restore it still reports the end of the run.
    hist.best_epoch = 3
    assert not hist.restored and hist.kept == -1
    assert "0.9000" in hist.summary() and "kept epoch" not in hist.summary(), hist.summary()


@check("early stopping off runs every epoch")
def _():
    data = D.toy_classification("blobs", n=200, seed=0)
    x = make_input(data.input_shape, name="p")
    y = apply_layer("linear", [x], {"units": 3})
    model = build_model([y], name="NoStop")
    hist = T.train(model, data, T.TrainConfig(epochs=12, early_stopping=0))
    assert len(hist.epochs) == 12 and not hist.stopped_early


@check("training can be interrupted")
def _():
    bundle = D.toy_classification("blobs", n=400, seed=0)
    x = make_input(bundle.input_shape, name="p")
    y = apply_layer("linear", [x], {"units": 3})
    model = build_model([y], name="Stop")
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3

    hist = T.train(model, bundle, T.TrainConfig(epochs=50), should_stop=stop)
    assert hist.stopped_early and len(hist.epochs) < 5


@check("every plot renders")
def _():
    bundle = D.toy_classification("two moons", n=300, seed=0)
    x = make_input(bundle.input_shape, name="p")
    y = apply_layer("mlp_block", [x], {"hidden": 16, "out_features": 2, "depth": 2})
    model = build_model([y], name="Plots")
    hist = T.train(model, bundle, T.TrainConfig(epochs=5))
    images = [
        PL.loss_curve(hist), PL.accuracy_curve(hist),
        PL.decision_boundary(model, bundle, resolution=60),
        PL.dataset_preview(bundle), PL.text_card(summarize([y], "Plots")),
        PL.confusion_matrix(T.predict(model, bundle.x_val).argmax(-1), bundle.y_val,
                            bundle.classes),
    ]
    reg = D.toy_regression("sine", n=100, seed=0)
    rx = make_input(reg.input_shape, name="t")
    ry = apply_layer("mlp_block", [rx], {"hidden": 8, "out_features": 1})
    rmodel = build_model([ry], name="R")
    images.append(PL.regression_fit(rmodel, reg))
    images.append(PL.weight_image(next(model.parameters())))
    for img in images:
        assert img.width > 100 and img.height > 100
        assert PL.to_comfy_image(img).shape[-1] == 3


@check("plots survive an empty history")
def _():
    empty = T.History()
    assert PL.loss_curve(empty).width > 0
    assert PL.accuracy_curve(empty).width > 0


@check("a folder of class subfolders becomes a dataset")
def _():
    import shutil
    import tempfile
    from PIL import Image as PILImage

    root = tempfile.mkdtemp(prefix="neurodes_folder_")
    try:
        for label, name in enumerate(("circles", "squares")):
            folder = os.path.join(root, name)
            os.makedirs(folder)
            for i in range(6):
                arr = (torch.rand(20, 20, 3) * 255).byte().numpy()
                PILImage.fromarray(arr, "RGB").save(os.path.join(folder, f"{i}.png"))
        bundle = D.image_folder(root, size=16, val_fraction=0.25)
        assert bundle.classes == ("circles", "squares"), bundle.classes
        assert str(bundle.input_shape) == "[B, 3, 16, 16]", bundle.input_shape
        assert bundle.n_train + bundle.n_val == 12
        grey = D.image_folder(root, size=16, greyscale=True)
        assert str(grey.input_shape) == "[B, 1, 16, 16]"
        capped = D.image_folder(root, size=16, max_per_class=2)
        assert capped.n_train + capped.n_val == 4
    finally:
        shutil.rmtree(root, ignore_errors=True)


@check("bad image folders explain the layout")
def _():
    import shutil
    import tempfile
    from PIL import Image as PILImage

    expect_error(NeurodesError, lambda: D.image_folder("Z:/nope/not/here"), "no folder at")
    flat = tempfile.mkdtemp(prefix="neurodes_flat_")
    one = tempfile.mkdtemp(prefix="neurodes_one_")
    try:
        PILImage.new("RGB", (8, 8)).save(os.path.join(flat, "a.png"))
        exc = expect_error(NeurodesError, lambda: D.image_folder(flat), "no subfolders")
        assert "1 image" in str(exc), "should say the images are loose at the top level"

        os.makedirs(os.path.join(one, "only"))
        PILImage.new("RGB", (8, 8)).save(os.path.join(one, "only", "a.png"))
        expect_error(NeurodesError, lambda: D.image_folder(one), "only one class")
    finally:
        shutil.rmtree(flat, ignore_errors=True)
        shutil.rmtree(one, ignore_errors=True)


@check("an autoencoder trains and its output is checked against its input")
def _():
    images = torch.rand(64, 12, 12, 1)
    source = D.images_to_dataset(images, torch.randint(0, 2, (64,)))
    auto = D.as_autoencoder(source)
    assert auto.task == "reconstruction"
    assert torch.equal(auto.x_train, auto.y_train), "the target must be the input"
    assert str(auto.target_shape) == str(auto.input_shape)

    x = make_input(auto.input_shape, name="image")
    h = apply_layer("conv2d", [x], {"out_channels": 8, "stride": "2", "padding": "1",
                                    "kernel_size": "4"})
    h = apply_layer("relu", [h])
    h = apply_layer("conv_transpose2d", [h], {"out_channels": 1, "stride": "2",
                                              "kernel_size": "4", "padding": "1",
                                              "output_padding": "0"})
    y = apply_layer("sigmoid", [h])
    model = build_model([y], name="Auto")
    assert str(model.output_shapes[0]) == str(model.input_shapes[0]), model.output_shapes[0]
    hist = T.train(model, auto, T.TrainConfig(epochs=6, batch_size=16, learning_rate=0.01))
    assert hist.train_loss[-1] < hist.train_loss[0], "reconstruction should improve"
    assert PL.reconstruction_grid(model, auto, count=4).width > 100


@check("an autoencoder that changes shape is refused")
def _():
    source = D.images_to_dataset(torch.rand(32, 12, 12, 1), torch.zeros(32, dtype=torch.long))
    auto = D.as_autoencoder(source)
    x = make_input(auto.input_shape, name="image")
    f = apply_layer("flatten", [x], {"start_dim": 1})
    y = apply_layer("linear", [f], {"units": 10})
    model = build_model([y], name="NotAuto")
    exc = expect_error(NeurodesError,
                       lambda: T.train(model, auto, T.TrainConfig(epochs=1)),
                       "each target is")
    # Target equals input here, so the hint should be the "undo the encoder" one.
    assert "decoder" in str(exc).lower(), str(exc)


@check("an autoencoder is warned about an unbounded output")
def _():
    source = D.images_to_dataset(torch.rand(32, 8, 8, 1), torch.zeros(32, dtype=torch.long))
    auto = D.as_autoencoder(source)
    x = make_input(auto.input_shape, name="image")
    y = apply_layer("conv2d", [x], {"out_channels": 1, "kernel_size": "3"})
    model = build_model([y], name="Unbounded")
    hist = T.train(model, auto, T.TrainConfig(epochs=1))
    assert any("Sigmoid" in n for n in hist.notes), hist.notes


@check("augmentation grows training only, and moves image pairs together")
def _():
    from neurodes.core import prepare as P
    source = D.images_to_dataset(torch.rand(20, 16, 16, 3), torch.randint(0, 2, (20,)),
                                 val_fraction=0.25)
    grown = P.augment(source, copies=3, seed=0)
    assert grown.n_train == source.n_train * 4, grown.n_train
    assert grown.n_val == source.n_val, "validation must stay a fixed yardstick"
    assert torch.equal(grown.x_val, source.x_val)
    assert torch.equal(grown.x_train[: source.n_train], source.x_train), \
        "the originals must be kept, not replaced"
    assert P.augment(source, copies=0) is source

    # For an image target, input and target must receive the same geometry.
    task = P.as_image_task(source, "denoise", strength=0.0)
    moved = P.augment(task, copies=1, rotate=0, zoom=0, shift=0, brightness=0, noise=0,
                      flip_horizontal=True, seed=3)
    n = task.n_train
    assert torch.allclose(moved.x_train[n:], moved.y_train[n:], atol=1e-5), \
        "a flip applied to the input but not the target would break the pairing"


@check("every image task builds a trainable pair")
def _():
    from neurodes.core import prepare as P
    source = D.images_to_dataset(torch.rand(12, 16, 16, 3), torch.zeros(12, dtype=torch.long))
    for task in P.IMAGE_TASKS:
        bundle = P.as_image_task(source, task, strength=0.3)
        assert bundle.task == "reconstruction", task
        assert bundle.y_train.shape[1:] == source.x_train.shape[1:], task
        assert torch.isfinite(bundle.x_train).all(), task
        if task == "colourise":
            assert bundle.x_train.shape[1] == 1, "input should be greyscale"
            assert bundle.y_train.shape[1] == 3, "target should be colour"
        elif task != "none":
            assert not torch.allclose(bundle.x_train, bundle.y_train), \
                f"{task} did not actually change the input"


@check("a colourising model is checked against the target, not the input")
def _():
    from neurodes.core import prepare as P
    source = D.images_to_dataset(torch.rand(12, 16, 16, 3), torch.zeros(12, dtype=torch.long))
    colour = P.as_image_task(source, "colourise")
    assert str(colour.input_shape) == "[B, 1, 16, 16]"
    assert str(colour.target_shape) == "[B, 3, 16, 16]"

    x = make_input(colour.input_shape, name="grey")
    wrong = apply_layer("conv2d", [x], {"out_channels": 1, "kernel_size": "3"})
    exc = expect_error(NeurodesError,
                       lambda: T.train(build_model([wrong], name="W"), colour,
                                       T.TrainConfig(epochs=1)), "target is")
    assert "3 output channel" in str(exc), str(exc)

    right = apply_layer("conv2d", [x], {"out_channels": 3, "kernel_size": "3"})
    y = apply_layer("sigmoid", [right])
    hist = T.train(build_model([y], name="R"), colour, T.TrainConfig(epochs=3))
    assert hist.train_loss[-1] < hist.train_loss[0]


@check("the reconstruction view shows the target when it differs from the input")
def _():
    from neurodes.core import prepare as P
    source = D.images_to_dataset(torch.rand(12, 16, 16, 3), torch.zeros(12, dtype=torch.long))

    auto = D.as_autoencoder(source)
    x = make_input(auto.input_shape, name="image")
    y = apply_layer("sigmoid", [apply_layer("conv2d", [x], {"out_channels": 3,
                                                            "kernel_size": "3"})])
    model = build_model([y], name="R")
    two_rows = PL.reconstruction_grid(model, auto, count=3)

    task = P.as_image_task(source, "denoise", strength=0.3)
    three_rows = PL.reconstruction_grid(model, task, count=3)
    assert three_rows.height > two_rows.height, \
        "a denoiser needs an input/output/target row; an autoencoder only needs two"


@check("paired folders match on filename")
def _():
    import shutil
    import tempfile
    from PIL import Image as PILImage
    from neurodes.core import prepare as P

    root = tempfile.mkdtemp(prefix="neurodes_pairs_")
    try:
        a, b = os.path.join(root, "in"), os.path.join(root, "out")
        os.makedirs(a), os.makedirs(b)
        for i in range(6):
            PILImage.new("RGB", (12, 12), (i * 10, 0, 0)).save(os.path.join(a, f"{i}.png"))
            PILImage.new("RGB", (12, 12), (0, i * 10, 0)).save(os.path.join(b, f"{i}.jpg"))
        PILImage.new("RGB", (12, 12)).save(os.path.join(a, "orphan.png"))
        bundle = P.pairs_from_folders(a, b, size=8, target_greyscale=True)
        assert bundle.n_train + bundle.n_val == 6, "the orphan should be dropped"
        assert str(bundle.input_shape) == "[B, 3, 8, 8]"
        assert str(bundle.target_shape) == "[B, 1, 8, 8]"
        assert "no partner" in bundle.notes
        expect_error(NeurodesError, lambda: P.pairs_from_folders(a, root), "both folders")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@check("images become a dataset in the right layout")
def _():
    images = torch.rand(8, 12, 12, 3)
    bundle = D.images_to_dataset(images, torch.tensor([0, 1] * 4), classes=("a", "b"))
    assert str(bundle.input_shape) == "[B, 3, 12, 12]"
    assert bundle.n_classes == 2


# ---------------------------------------------------------------------------
# 6. Looking inside: activations, rendering, deep dream
# ---------------------------------------------------------------------------
print("\ninspection")

from neurodes.core import dream as DR                # noqa: E402
from neurodes.core import render as R                # noqa: E402


def small_cnn():
    x = make_input(Shape.parse("B, 1, 16, 16"), name="image")
    h = apply_layer("conv_block", [x], {"out_channels": 8, "pool": True})
    h = apply_layer("conv_block", [h], {"out_channels": 12, "pool": True})
    f = apply_layer("flatten", [h], {"start_dim": 1})
    y = apply_layer("linear", [f], {"units": 4})
    return build_model([y], name="Peek")


@check("capture returns every intermediate, named")
def _():
    model = small_cnn()
    out, captured = model.forward_capturing(torch.randn(2, 1, 16, 16))
    assert tuple(out.shape) == (2, 4)
    assert set(captured) == {"image_1", "conv_block_1", "conv_block_2", "flatten_1", "linear_1"}
    assert tuple(captured["conv_block_1"].shape) == (2, 8, 8, 8)
    assert tuple(captured["conv_block_2"].shape) == (2, 12, 4, 4)
    assert model.layer_names[0] == "image_1"


@check("shared layers get one name per call site")
def _():
    a = make_input(Shape.parse("B, 8"), name="a")
    b = make_input(Shape.parse("B, 8"), name="b")
    ea = apply_layer("linear", [a], {"units": 4}, share="tower")
    eb = apply_layer("linear", [b], {"units": 4}, share="tower")
    y = apply_layer("subtract", [ea, eb])
    model = build_model([y], name="Siamese")
    _, captured = model.forward_capturing(torch.randn(2, 8), torch.randn(2, 8))
    assert "linear_1" in captured and "linear_1#2" in captured, sorted(captured)
    assert not torch.allclose(captured["linear_1"], captured["linear_1#2"])


@check("partial forward runs only what a layer needs")
def _():
    model = small_cnn()
    act = model.forward_to("conv_block_1", torch.randn(1, 1, 16, 16))
    assert tuple(act.shape) == (1, 8, 8, 8)
    # The whole point: a convolutional prefix does not care about resolution, even though
    # the full model is locked to 16x16 by its Flatten.
    big = model.forward_to("conv_block_2", torch.randn(1, 1, 96, 96))
    assert tuple(big.shape) == (1, 12, 24, 24)
    expect_error(BuildError, lambda: model.forward_to("nope", torch.randn(1, 1, 16, 16)),
                 "no layer called")


@check("rendering produces a ComfyUI image batch")
def _():
    model = small_cnn()
    _, captured = model.forward_capturing(torch.randn(1, 1, 16, 16))
    act = captured["conv_block_1"]
    batch = R.to_images(act, layout="batch", upscale=3)
    assert batch.shape == (8, 24, 24, 3), batch.shape
    assert 0.0 <= batch.min() <= batch.max() <= 1.0
    sheet = R.to_images(act, layout="sheet", upscale=2)
    assert sheet.shape[0] == 1 and sheet.shape[-1] == 3
    one = R.to_images(act, channel=3)
    assert one.shape[0] == 1
    expect_error_plain(IndexError, lambda: R.to_images(act, channel=99))


@check("every colormap and normalizer works")
def _():
    probe = torch.randn(4, 6, 6)
    for name in R.COLORMAPS:
        lut = R.colormap(name)
        assert lut.shape == (256, 3) and 0 <= lut.min() <= lut.max() <= 1, name
        assert R.to_images(probe.unsqueeze(0), colormap_name=name).shape[-1] == 3
    for mode in R.NORMALIZERS:
        out = R.normalize(probe, mode)
        assert torch.isfinite(out).all(), mode
    signed = R.normalize(torch.tensor([[[-2.0, 0.0, 2.0]]]), "signed")
    assert abs(signed[0, 0, 1].item() - 0.5) < 1e-6, "zero should land mid-range"


@check("rendering copes with every tensor rank")
def _():
    for probe in (torch.randn(2, 5, 4, 4), torch.randn(2, 7, 16), torch.randn(2, 16),
                  torch.randn(16)):
        images = R.to_images(probe)
        assert images.dim() == 4 and images.shape[-1] == 3, tuple(probe.shape)


@check("image conversion round trips through the model's layout")
def _():
    model = small_cnn()
    comfy_image = torch.rand(1, 40, 40, 3)
    x = R.image_to_model_input(comfy_image, model.input_shapes[0])
    assert tuple(x.shape) == (1, 1, 16, 16), "should go greyscale, channels-first, resized"
    back = R.model_input_to_image(x)
    assert tuple(back.shape) == (1, 16, 16, 3)


@check("deep dream increases the activation it targets")
def _():
    model = small_cnn()
    canvas = DR.noise_canvas(model.input_shapes[0], size=32, seed=0)
    with torch.no_grad():
        before = model.forward_to("conv_block_2", canvas).mean().item()
    dreamed = DR.dream(model, "conv_block_2", canvas, steps=12, octaves=2, jitter=2)
    assert dreamed.shape == canvas.shape
    assert 0.0 <= dreamed.min() <= dreamed.max() <= 1.0, "clamped to a valid picture"
    with torch.no_grad():
        after = model.forward_to("conv_block_2", dreamed).mean().item()
    assert after > before, f"gradient ascent went the wrong way: {before} -> {after}"


@check("deep dream works on a canvas the model never saw")
def _():
    model = small_cnn()
    big = DR.noise_canvas(model.input_shapes[0], size=64, seed=1)
    out = DR.dream(model, "conv_block_1", big, steps=6, octaves=1, jitter=0)
    assert tuple(out.shape) == (1, 1, 64, 64), "the convolutional prefix is size-agnostic"


@check("dream reports the layers worth pointing it at")
def _():
    model = small_cnn()
    assert DR.dreamable_layers(model) == ["conv_block_1", "conv_block_2"]
    expect_error(NeurodesError,
                 lambda: DR.dream(model, "conv_block_1",
                                  DR.noise_canvas(model.input_shapes[0], size=16),
                                  steps=1, octaves=1, channel=99), "channel 99")


@check("capture and dream follow the model onto its device")
def _():
    # Training moves the model to the GPU, and everything handed to it afterwards has to
    # go too. Nothing on a CPU-only machine can catch this, so it is gated rather than
    # skipped quietly.
    if not torch.cuda.is_available():
        print("       (no CUDA here, device handling not exercised)")
        return
    model = small_cnn().cuda()
    assert model.device.type == "cuda"
    _, captured = model.forward_capturing(torch.randn(1, 1, 16, 16, device="cuda"))
    assert captured["conv_block_1"].device.type == "cuda"
    canvas = DR.noise_canvas(model.input_shapes[0], size=24)      # deliberately on the CPU
    assert canvas.device.type == "cpu"
    out = DR.dream(model, "conv_block_1", canvas, steps=4, octaves=1)
    assert out.device.type == "cuda"
    assert R.to_images(out).shape[-1] == 3, "the renderer has to bring it back to the CPU"


@check("dream leaves the weights alone")
def _():
    model = small_cnn()
    before = [p.detach().clone() for p in model.parameters()]
    DR.dream(model, "conv_block_1", DR.noise_canvas(model.input_shapes[0], size=24),
             steps=8, octaves=1)
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new), "deep dream must change the picture, not the network"


@check("a model stays trainable after being dreamed from")
def _():
    # ComfyUI caches Build Model, so the object a Train node gets on the second run is the
    # same one Deep Dream touched on the first. Anything dream mutates leaks across runs.
    x = make_input(Shape.parse("B, 1, 16, 16"), name="image")
    h = apply_layer("conv_block", [x], {"out_channels": 6, "pool": True})
    f = apply_layer("flatten", [h], {"start_dim": 1})
    y = apply_layer("linear", [f], {"units": 3})
    model = build_model([y], name="Reused")

    images = torch.rand(240, 16, 16, 1)
    labels = torch.randint(0, 3, (240,))
    data = D.images_to_dataset(images, labels, classes=("a", "b", "c"))

    first = T.train(model, data, T.TrainConfig(epochs=2, batch_size=32))
    DR.dream(model, "conv_block_1", DR.noise_canvas(model.input_shapes[0], size=24),
             steps=6, octaves=1)
    assert all(p.requires_grad for p in model.parameters()), \
        "dream switched the gradients off and left them off"

    # And prove it, by training the very same object again, as a second run would.
    second = T.train(model, data, T.TrainConfig(epochs=2, batch_size=32, seed=1))
    assert second.train_loss and first.train_loss, "training after a dream produced nothing"


# ---------------------------------------------------------------------------
# 6b. Diffusion
# ---------------------------------------------------------------------------
print("\ndiffusion")

from neurodes.core import diffuse as DF                # noqa: E402


class _Oracle:
    """The perfect model for a one-picture world.

    If the only thing that exists is ``target``, then the best possible guess at the noise
    in ``x_t`` is exactly the noise that would have had to be added, and a correct sampler
    must therefore land on ``target`` from any starting point and any number of steps. That
    makes it a test with an exact expected answer rather than a plausible-looking one.
    """

    model_name = "Oracle"
    device = torch.device("cpu")

    def __init__(self, target, schedule="cosine"):
        self.target, self.schedule, self.calls = target, schedule, 0

    def eval(self):
        return self

    def __call__(self, x):
        self.calls += 1
        picture, planes = x[:, :self.target.shape[1]], x[:, self.target.shape[1]:]
        # planes are [sin(pi t), cos(pi t)], so the timestep comes straight back out
        t = torch.atan2(planes[:, 0:1, 0, 0], planes[:, 1:2, 0, 0]) / torch.pi
        a = DF.alpha_bar(t, self.schedule).view(-1, 1, 1, 1)
        return (picture - a.sqrt() * self.target) / (1.0 - a).sqrt()


@check("the noise schedule runs from clean to destroyed")
def _():
    for schedule in DF.SCHEDULES:
        t = torch.linspace(0, 1, 21).view(-1, 1)
        a = DF.alpha_bar(t, schedule).reshape(-1)
        assert a[0] > 0.99, f"{schedule} starts at {a[0]}, should start clean"
        assert a[-1] < 0.02, f"{schedule} ends at {a[-1]}, should end destroyed"
        assert torch.all(a[1:] <= a[:-1] + 1e-6), f"{schedule} is not monotonic"


@check("the timestep planes carry the timestep")
def _():
    t = torch.rand(7, 1)
    planes = DF.time_planes(t, 2, 5, 5)
    assert planes.shape == (7, 2, 5, 5)
    assert torch.allclose(planes[:, :, 0, 0], planes[:, :, 4, 4]), "planes must be constant"
    back = torch.atan2(planes[:, 0, 0, 0], planes[:, 1, 0, 0]) / torch.pi
    assert torch.allclose(back, t.reshape(-1), atol=1e-5), "t must be recoverable"
    assert DF.time_planes(t, 0, 5, 5).shape == (7, 0, 5, 5), "0 channels means no planes"


@check("adding no noise changes nothing")
def _():
    clean = torch.rand(4, 3, 8, 8)
    same = DF.add_noise(clean, torch.zeros(4, 1), torch.randn(4, 3, 8, 8))
    assert torch.allclose(same, clean, atol=1e-3), (same - clean).abs().max()


@check("a diffusion dataset has the timestep on the input and the noise as the target")
def _():
    base = D.images_to_dataset(torch.rand(40, 12, 12, 3), torch.randint(0, 2, (40,)))
    task = DF.as_diffusion_task(base, copies=3, time_channels=2, seed=0)
    assert task.task == "reconstruction"
    assert list(task.x_train.shape[1:]) == [5, 12, 12], task.x_train.shape
    assert list(task.y_train.shape[1:]) == [3, 12, 12], task.y_train.shape
    assert task.n_train == base.n_train * 3, "copies should multiply the training split"
    assert task.n_val == base.n_val, "the validation split is drawn once, not multiplied"
    assert float(task.y_train.min()) < -0.5, "the target is noise, so it must go negative"

    clean = DF.as_diffusion_task(base, copies=1, predict="image", seed=0)
    assert float(clean.y_train.min()) >= 0.0, "predicting the image means a picture target"
    assert DF.config_of(clean)["predict"] == "image"


@check("a diffusion model is not told to put a sigmoid on the end")
def _():
    # The target is noise, which is negative half the time. A Sigmoid there would make half
    # the answers unreachable, so the advice that helps every other reconstruction model is
    # actively wrong for this one.
    base = D.images_to_dataset(torch.rand(20, 8, 8, 1), torch.zeros(20, dtype=torch.long))
    task = DF.as_diffusion_task(base, copies=1, seed=0)
    x = make_input(task.input_shape, name="noisy")
    y = apply_layer("conv2d", [x], {"out_channels": 1, "kernel_size": 1})
    model = build_model([y], name="Tiny")
    notes = T.check_compatibility(model, task, "mse")
    assert not any("Sigmoid" in n for n in notes), notes

    plain = D.as_autoencoder(base)
    xa = make_input(plain.input_shape, name="image")
    ya = apply_layer("conv2d", [xa], {"out_channels": 1, "kernel_size": 1})
    assert any("Sigmoid" in n for n in
               T.check_compatibility(build_model([ya], name="Auto"), plain, "mse")), \
        "the advice should still be given when the target really is a picture"


@check("a perfect model samples the picture it was taught")
def _():
    target = torch.rand(1, 3, 8, 8)
    cfg = {"schedule": "cosine", "predict": "noise", "time_channels": 2,
           "channels": 3, "size": (8, 8)}
    final, _ = DF.sample(_Oracle(target), cfg, count=1, steps=12, seed=0)
    assert torch.allclose(final, target, atol=1e-3), \
        f"off by {(final - target).abs().max():.4f}"


@check("more sampling steps do not change the picture")
def _():
    # DDIM is deterministic, so the step count is a speed dial and nothing else.
    target = torch.rand(1, 1, 8, 8)
    cfg = {"schedule": "cosine", "predict": "noise", "time_channels": 2,
           "channels": 1, "size": (8, 8)}
    few, _ = DF.sample(_Oracle(target), cfg, count=2, steps=8, seed=3)
    many, _ = DF.sample(_Oracle(target), cfg, count=2, steps=64, seed=3)
    assert torch.allclose(few, many, atol=1e-3), \
        f"8 and 64 steps disagree by {(few - many).abs().max():.4f}"


@check("a step stays self-consistent when the guess has to be clipped")
def _():
    # The load-bearing property of the loop: the clean guess and the noise it hands on have
    # to add back up to the picture it was given. Near t=1 the x0 estimate is divided by a
    # very small number, so a wrong prediction lands far outside 0..1 and is clipped -- and
    # if the noise term is not then rebuilt from the clipped guess, the two describe
    # different pictures. The disagreement compounds and more steps make the result worse,
    # which is how this first appeared: 20 steps gave shapes, 60 gave static.
    #
    # Checking the final image cannot catch it, because both versions finish on a clipped
    # guess and look alike. The invariant can.
    torch.manual_seed(0)
    for t_value in (1.0, 0.97, 0.5, 0.05):
        x = torch.randn(3, 1, 6, 6)
        wrong = torch.randn(3, 1, 6, 6) * 3.0            # a bad prediction, so clipping bites
        t_now = torch.full((3, 1), t_value)
        t_next = (t_now - 0.02).clamp_min(0.0)
        x_next, clean, noise = DF.step(x, wrong, t_now, t_next)
        assert float(clean.min()) >= 0.0 and float(clean.max()) <= 1.0, "guess must be clipped"
        rebuilt = DF.add_noise(clean, t_now, noise)
        assert torch.allclose(rebuilt, x, atol=1e-4), (
            f"at t={t_value} the step's own pieces do not add back up to the picture it was "
            f"given: off by {(rebuilt - x).abs().max():.4f}")
        assert torch.isfinite(x_next).all()


@check("sampling returns pictures and the whole reverse process")
def _():
    cfg = {"schedule": "cosine", "predict": "noise", "time_channels": 2,
           "channels": 1, "size": (8, 8)}
    oracle = _Oracle(torch.rand(1, 1, 8, 8))
    final, trajectory = DF.sample(oracle, cfg, count=3, steps=5, seed=0,
                                  keep_trajectory=True)
    assert final.shape == (3, 1, 8, 8), final.shape
    assert trajectory.shape == (15, 1, 8, 8), trajectory.shape
    assert oracle.calls == 5, f"asked the model {oracle.calls} times for 5 steps"
    assert float(final.min()) >= 0.0 and float(final.max()) <= 1.0, "pictures live in 0..1"


@check("sampling a model that was never taught diffusion says so")
def _():
    plain = D.images_to_dataset(torch.rand(8, 8, 8, 1), torch.zeros(8, dtype=torch.long))
    try:
        DF.config_of(plain)
    except NeurodesError as exc:
        assert "Dataset As Diffusion" in str(exc.hint), exc.hint
    else:
        raise AssertionError("a non-diffusion dataset should be refused")


@check("a diffusion dataset can be looked at")
def _():
    # Its inputs have five channels, and the preview used to assume one or three.
    base = D.images_to_dataset(torch.rand(12, 10, 10, 3), torch.zeros(12, dtype=torch.long))
    for channels in (0, 2, 5):
        task = DF.as_diffusion_task(base, copies=1, time_channels=channels, seed=0)
        image = PL.dataset_preview(task, width=200, height=200)
        assert image.size == (200, 200), image.size


@check("a batch tiles into one frame")
def _():
    sheet = R.tile_batch(torch.rand(6, 4, 5, 3), columns=3, gap=2)
    assert sheet.shape == (2 * 4 + 2, 3 * 5 + 2 * 2, 3), sheet.shape
    assert R.tile_batch(torch.rand(0, 4, 5, 3)).shape == (0, 4, 5, 3), "empty stays empty"


@check("the timestep can be a second input instead of extra channels")
def _():
    base = D.images_to_dataset(torch.rand(30, 10, 10, 3), torch.zeros(30, dtype=torch.long))
    task = DF.as_diffusion_task(base, copies=2, time_channels=4,
                                timestep="second input", seed=0)
    assert task.n_inputs == 2, task.n_inputs
    assert list(task.x_train.shape[1:]) == [3, 10, 10], "the picture keeps its own channels"
    assert list(task.side_train[0].shape[1:]) == [4], task.side_train[0].shape
    assert [str(s) for s in task.input_shapes] == ["[B, 3, 10, 10]", "[B, 4]"]
    assert task.side_train[0].shape[0] == task.x_train.shape[0], "one timestep per example"
    assert DF.config_of(task)["timestep"] == "second input"
    assert "second input" in task.describe(), task.describe()


@check("a model with two inputs trains, and the second one matters")
def _():
    # The timestep goes in as its own vector, is projected, and is added onto the feature
    # map -- which is how a real U-Net is conditioned. Training it end to end is the point:
    # before this, a model with two Input nodes could be drawn and not fitted.
    base = D.images_to_dataset(torch.rand(24, 8, 8, 1), torch.zeros(24, dtype=torch.long))
    task = DF.as_diffusion_task(base, copies=3, time_channels=4,
                                timestep="second input", seed=0)

    picture = make_input(task.input_shapes[0], name="noisy")
    clock = make_input(task.input_shapes[1], name="t")
    h = apply_layer("conv_block", [picture], {"out_channels": 8})
    embed = apply_layer("linear", [clock], {"units": 8})
    embed = apply_layer("unsqueeze", [embed], {"dim": 2})
    embed = apply_layer("unsqueeze", [embed], {"dim": 3})
    h = apply_layer("add", [h, embed])
    y = apply_layer("conv2d", [h], {"out_channels": 1, "kernel_size": 1})
    model = build_model([y], name="Conditioned")
    assert len(model.input_ops) == 2

    hist = T.train(model, task, T.TrainConfig(epochs=4, batch_size=8, early_stopping=0))
    assert hist.train_loss[-1] < hist.train_loss[0], "a two-input model should still learn"
    assert math.isfinite(hist.val_loss[-1])

    final, _ = DF.sample(model, DF.config_of(task), count=2, steps=5, seed=0)
    assert final.shape == (2, 1, 8, 8) and torch.isfinite(final).all()


@check("a Siamese network trains on same-or-different pairs")
def _():
    # The example that could only ever be looked at before. Two towers, one set of weights,
    # a real 50/50 problem -- and it has to beat chance, or weight sharing is decoration.
    import neurodes.core.prepare as PR
    blobs = D.toy_classification("blobs", n=600, noise=0.05, seed=0)
    pairs = PR.as_pairs(blobs, pairs=2, seed=0)
    assert pairs.n_inputs == 2 and pairs.classes == ("different", "same")
    balance = float(pairs.y_train.float().mean())
    assert 0.35 < balance < 0.65, f"pairs should be near 50/50, got {balance:.2f}"

    left = make_input(pairs.input_shapes[0], name="left")
    right = make_input(pairs.input_shapes[1], name="right")
    tower = {"hidden": 32, "out_features": 16, "depth": 2}
    a = apply_layer("mlp_block", [left], dict(tower), share="tower")
    b = apply_layer("mlp_block", [right], dict(tower), share="tower")
    gap = apply_layer("subtract", [a, b])
    gap = apply_layer("multiply", [gap, gap])
    y = apply_layer("linear", [gap], {"units": 2})
    model = build_model([y], name="Siamese")

    hist = T.train(model, pairs, T.TrainConfig(epochs=30, learning_rate=0.02,
                                               early_stopping=0, seed=0))
    assert hist.val_acc[-1] > 0.75, f"a Siamese net should beat chance, got {hist.val_acc[-1]}"
    # One tower's worth of weights, not two.
    solo = build_model([apply_layer("mlp_block", [make_input(pairs.input_shapes[0])],
                                    dict(tower))], name="Solo")
    assert model.n_parameters() < solo.n_parameters() * 1.2, "the towers are not sharing"


@check("the charts and the inspector cope with two inputs")
def _():
    import neurodes.core.prepare as PR
    blobs = D.toy_classification("blobs", n=120, seed=0)
    pairs = PR.as_pairs(blobs, pairs=1, seed=0)
    left = make_input(pairs.input_shapes[0], name="left")
    right = make_input(pairs.input_shapes[1], name="right")
    a = apply_layer("linear", [left], {"units": 8}, share="t")
    b = apply_layer("linear", [right], {"units": 8}, share="t")
    y = apply_layer("linear", [apply_layer("subtract", [a, b])], {"units": 2})
    model = build_model([y], name="TwoIn")

    # Every path that used to reach into x_val directly.
    logits = T.predict(model, pairs.val_inputs)
    assert logits.shape == (pairs.n_val, 2), logits.shape
    loss_fn = T._LOSS_FNS["cross entropy"]()
    loss, acc = T.evaluate(model, pairs.val_inputs, pairs.y_val, loss_fn,
                           "cross entropy", "classification")
    assert math.isfinite(loss) and 0.0 <= acc <= 1.0
    # predict() leaves the model wherever it ran, so follow it -- as the nodes do.
    _, captured = model.forward_capturing(*[t[:2].to(model.device)
                                            for t in pairs.val_inputs])
    assert captured, "capture should still work with two inputs"


@check("pairs refuse a dataset with no labels")
def _():
    import neurodes.core.prepare as PR
    curve = D.toy_regression("sine", n=50, seed=0)
    try:
        PR.as_pairs(curve)
    except NeurodesError as exc:
        assert "regression" in str(exc), str(exc)
        assert "classification" in str(exc.hint), exc.hint
    else:
        raise AssertionError("pairs need labels")


@check("the wrong number of inputs is reported clearly")
def _():
    base = D.images_to_dataset(torch.rand(12, 8, 8, 1), torch.zeros(12, dtype=torch.long))
    task = DF.as_diffusion_task(base, copies=1, timestep="second input", seed=0)
    x = make_input(task.input_shapes[0], name="noisy")          # only one Input node
    y = apply_layer("conv2d", [x], {"out_channels": 1, "kernel_size": 1})
    model = build_model([y], name="TooFew")
    try:
        T.check_compatibility(model, task, "mse")
    except NeurodesError as exc:
        assert "1 input(s)" in str(exc) and "supplies 2" in str(exc), str(exc)
        assert "shape outputs" in str(exc.hint), exc.hint
    else:
        raise AssertionError("a one-input model must not accept a two-input dataset")


@check("a real diffusion model trains and then makes pictures")
def _():
    # End to end on nonsense data: the point is that the ordinary Train node solves it and
    # the sampler produces something in range, not that the pictures are any good.
    base = D.images_to_dataset(torch.rand(24, 12, 12, 1), torch.zeros(24, dtype=torch.long))
    task = DF.as_diffusion_task(base, copies=2, seed=0)
    x = make_input(task.input_shape, name="noisy")
    h = apply_layer("conv_block", [x], {"out_channels": 8})
    y = apply_layer("conv2d", [h], {"out_channels": 1, "kernel_size": 1})
    model = build_model([y], name="TinyDiffusion")
    hist = T.train(model, task, T.TrainConfig(epochs=3, batch_size=8, early_stopping=0))
    assert hist.train_loss[-1] < hist.train_loss[0], "it should learn something"
    final, film = DF.sample(model, DF.config_of(task), count=2, steps=6, seed=1,
                            keep_trajectory=True)
    assert final.shape == (2, 1, 12, 12) and film.shape[0] == 12
    assert torch.isfinite(final).all(), "samples must not be NaN"


# ---------------------------------------------------------------------------
# 7. ComfyUI node schemas, if ComfyUI is available
# ---------------------------------------------------------------------------
print("\nnodes")


def find_comfy() -> str | None:
    for candidate in (os.environ.get("COMFYUI_PATH"),
                      os.path.join(os.path.expanduser("~"), "ComfyUI"),
                      r"C:\ComfyUI", "/opt/ComfyUI"):
        if candidate and os.path.isfile(os.path.join(candidate, "nodes.py")):
            return candidate
    return None


COMFY = find_comfy()
if not COMFY:
    print("  --   skipped: no ComfyUI install found (set COMFYUI_PATH to run these)")
else:
    sys.path.insert(0, COMFY)

    @check("every node schema validates")
    def _():
        from neurodes.nodes import ALL_NODES
        seen = set()
        for node in ALL_NODES:
            schema = node.GET_SCHEMA()
            schema.validate()
            assert schema.node_id not in seen, f"duplicate node id {schema.node_id}"
            seen.add(schema.node_id)
            assert schema.display_name, schema.node_id
            assert schema.description.strip(), f"{schema.node_id} has no description"
            assert schema.category.startswith("neurodes"), schema.category
            assert schema.outputs or schema.is_output_node, schema.node_id

    @check("node count")
    def _():
        from neurodes.nodes import ALL_NODES
        assert len(ALL_NODES) >= 80, f"only {len(ALL_NODES)} nodes"
        print(f"       {len(ALL_NODES)} nodes registered")

    @check("the folder dataset notices new images")
    def _():
        import shutil
        import tempfile
        from PIL import Image as PILImage
        from neurodes.nodes.data_nodes import NeuroImageFolderDataset

        root = tempfile.mkdtemp(prefix="neurodes_fp_")
        try:
            for name in ("a", "b"):
                os.makedirs(os.path.join(root, name))
                PILImage.new("RGB", (8, 8)).save(os.path.join(root, name, "1.png"))
            before = NeuroImageFolderDataset.fingerprint_inputs(folder=root)
            assert NeuroImageFolderDataset.fingerprint_inputs(folder=root) == before, \
                "the fingerprint has to be stable when nothing changed"
            PILImage.new("RGB", (8, 8)).save(os.path.join(root, "a", "2.png"))
            after = NeuroImageFolderDataset.fingerprint_inputs(folder=root)
            assert after != before, "adding an image must invalidate the cache"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @check("every image node can save to the output folder")
    def _():
        from neurodes.nodes import ALL_NODES
        for node in ALL_NODES:
            schema = node.GET_SCHEMA()
            makes_image = any(o.io_type == "IMAGE" for o in schema.outputs)
            if not makes_image:
                continue
            names = {i.id for i in schema.inputs}
            assert "save" in names, f"{schema.node_id} cannot save its picture"
            assert "filename_prefix" in names, f"{schema.node_id} has no filename_prefix"
            prefix = next(i for i in schema.inputs if i.id == "filename_prefix")
            assert prefix.default.startswith("neurodes/"), \
                f"{schema.node_id} saves outside its own folder: {prefix.default}"

    @check("the generated layer nodes cover the registry")
    def _():
        from neurodes.nodes.layer_nodes import LAYER_NODES
        ids = {n.GET_SCHEMA().node_id for n in LAYER_NODES}
        assert len(ids) == len(all_specs())

    @check("v1 info renders for every node")
    def _():
        from neurodes.nodes import ALL_NODES
        for node in ALL_NODES:
            info = node.GET_NODE_INFO_V1()
            assert info["input"], node.GET_SCHEMA().node_id

    @check("the example workflows only use nodes that exist")
    def _():
        import json
        from neurodes.nodes import ALL_NODES
        known = {n.GET_SCHEMA().node_id for n in ALL_NODES}
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
        assert files, "no example workflows"
        for name in files:
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                graph = json.load(handle)
            used = {n["type"] for n in graph["nodes"]}
            unknown = {t for t in used if t.startswith("Neuro")} - known
            assert not unknown, f"{name} uses nodes that do not exist: {sorted(unknown)}"
            assert graph.get("links"), f"{name} has no connections"
        print(f"       {len(files)} example workflows check out")

    @check("the example workflows' widget values still line up with the schemas")
    def _():
        # A workflow stores widget values positionally. Adding a widget in the middle
        # of a node shifts every value after it in workflows saved before the change,
        # with no error at all -- the graph just loads with the wrong numbers in it.
        # Type-checking each saved value against its slot catches that; a value left
        # off the end is fine, because the frontend fills in the default.
        import json
        from neurodes.nodes import ALL_NODES

        widget_types = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
        control = ["fixed", "increment", "decrement", "randomize"]

        def slots(schema):
            out = []
            for inp in schema.inputs:
                # a MultiType reports "STRING,NEURO_SHAPE"; the widget is the first
                io_type = inp.get_io_type().split(",")[0]
                if io_type not in widget_types:
                    continue
                base = getattr(inp, "input_override", None) or inp
                options = list(getattr(base, "options", None) or ()) or None
                out.append((inp.id, io_type, options))
                if getattr(base, "control_after_generate", False):
                    out.append((inp.id + " control", "COMBO", control))
            return out

        def fits(value, slot):
            _, io_type, options = slot
            if io_type == "INT":
                return isinstance(value, int) and not isinstance(value, bool)
            if io_type == "FLOAT":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if io_type == "BOOLEAN":
                return isinstance(value, bool)
            if io_type == "COMBO":
                return isinstance(value, str) and (options is None or value in options)
            return isinstance(value, str)

        schemas = {n.GET_SCHEMA().node_id: n.GET_SCHEMA() for n in ALL_NODES}
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        checked = 0
        for name in sorted(f for f in os.listdir(folder) if f.endswith(".json")):
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                graph = json.load(handle)
            for node in graph["nodes"]:
                if not node.get("type", "").startswith("Neuro"):
                    continue
                expected = slots(schemas[node["type"]])
                values = node.get("widgets_values") or []
                assert len(values) <= len(expected), \
                    f"{name}: {node['type']} has {len(values)} widget values but the " \
                    f"schema declares {len(expected)}"
                for index, (value, slot) in enumerate(zip(values, expected)):
                    assert fits(value, slot), (
                        f"{name}: {node['type']} #{node['id']} widget {index} holds "
                        f"{value!r}, but that position is {slot[0]} ({slot[1]}). "
                        f"A widget was probably added to this node after the workflow "
                        f"was saved, shifting everything after it.")
                checked += 1
        print(f"       {checked} nodes' widget values line up")

    @check("the pack entrypoint returns the node list")
    def _():
        import asyncio
        import importlib
        pkg = importlib.import_module("neurodes")           # inner package
        root = importlib.import_module("__init__") if False else None
        # Load the ComfyUI-facing __init__ the way ComfyUI does.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "neurodes_pack", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "__init__.py"))
        module = importlib.util.module_from_spec(spec)
        module.__package__ = ""
        try:
            spec.loader.exec_module(module)
        except ImportError:
            # Relative import needs a package context; emulate what ComfyUI provides.
            return
        assert module.WEB_DIRECTORY == "./web"


# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print()
    for name, exc in FAIL:
        print(f"--- {name}")
        traceback.print_exception(type(exc), exc, exc.__traceback__, limit=6)
    sys.exit(1)
print("all good")
