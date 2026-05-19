"""
Test: verify D stays in basis frame through ML pipeline operations.
Run: python tests/test_ml_basis.py
"""

from vdr import VDR
from vdr.linalg import Vec, Mat
from vdr.basis import set_default, get_default, to_qbasis, vec_to_qbasis, mat_to_qbasis

from vdr.basis import to_qbasis

BITS = 32
DENOM = None
passed = 0
failed = 0
total = 0


def setup():
    global DENOM
    set_default(BITS)
    DENOM = 2 ** get_default()
    print("basis: 2^%d = %d\n" % (BITS, DENOM))


def check(label, val, expect_d=None):
    global passed, failed, total
    total += 1
    if expect_d is None:
        expect_d = DENOM

    if isinstance(val, VDR):
        d = val.d
    elif isinstance(val, Vec):
        d = val[0].d
    elif isinstance(val, Mat):
        d = val[0, 0].d
    else:
        print("  %-50s SKIP (unknown type)" % label)
        return

    ok = d == expect_d
    tag = "OK" if ok else "FAIL D=%d" % d
    print("  %-50s [%s]" % (label, tag))
    if ok:
        passed += 1
    else:
        failed += 1


def check_all_vdr(label, vecs, expect_d=None):
    """Check that every VDR element in a list of Vec has correct D."""
    global passed, failed, total
    total += 1
    if expect_d is None:
        expect_d = DENOM

    bad = 0
    for v in vecs:
        for i in range(len(v)):
            if v[i].d != expect_d:
                bad += 1

    if bad == 0:
        print("  %-50s [OK]" % label)
        passed += 1
    else:
        print("  %-50s [FAIL %d bad elements]" % (label, bad))
        failed += 1


def test_nn():
    print("=== nn.py: Linear, ReLU, Sequential ===")
    from vdr.ml.nn import Linear, ReLU, Sequential, FFN

    layer = Linear.from_ints([[1, 2], [3, 4]], [0, 0])
    layer.to_qbasis(BITS)
    check("Linear weight after to_qbasis", layer.weight.value)
    check("Linear bias after to_qbasis", layer.bias.value)

    x = vec_to_qbasis(Vec.from_ints([1, 1]))
    y = layer.forward(x)
    check("Linear forward output", y)

    from vdr.ml.losses import mse_grad
    target = vec_to_qbasis(Vec.from_ints([0, 0]))
    grad = mse_grad(y, target)
    check("mse_grad output", grad)

    grad_in = layer.backward(grad)
    check("Linear backward grad_input", grad_in)
    check("Linear weight.grad after backward", layer.weight.grad)
    check("Linear bias.grad after backward", layer.bias.grad)

    act = ReLU()
    r = act.forward(y)
    check("ReLU forward output", r)
    rg = act.backward(grad)
    check("ReLU backward output", rg)

    model = Sequential([
        Linear.from_ints([[1, 2], [3, 4]], [0, 0]),
        ReLU(),
        Linear.from_ints([[1, 0], [0, 1]], [1, 1]),
    ])
    model.to_qbasis(BITS)
    out = model.forward(x)
    check("Sequential forward output", out)
    gout = model.backward(grad)
    check("Sequential backward output", gout)

    ffn = FFN(
        Linear.from_ints([[1, 2], [3, 4]], [0, 0]),
        ReLU(),
        Linear.from_ints([[1, 0], [0, 1]], [1, 1]),
    )
    ffn.to_qbasis(BITS)
    fout = ffn.forward(x)
    check("FFN forward output", fout)
    print()


def test_softmax():
    print("=== softmax.py: softmax, surrogate ===")
    from vdr.ml.softmax import softmax, softmax_surrogate_square

    logits = vec_to_qbasis(Vec.from_ints([1, 2, 3]))

    probs = softmax(logits, exp_depth=8)
    check("softmax output[0]", probs[0])

    total_p = to_qbasis(0)
    for i in range(len(probs)):
        total_p = total_p + probs[i]
    err = abs(total_p.to_float() - 1.0)
    print("  %-50s err=%.2e [%s]" % (
        "softmax sums to ~1",
        err,
        "OK" if err < 1e-6 else "FAIL"))

    probs2 = softmax_surrogate_square(logits)
    check("surrogate output[0]", probs2[0])

    total_s = to_qbasis(0)
    for i in range(len(probs2)):
        total_s = total_s + probs2[i]
    err2 = abs(total_s.to_float() - 1.0)
    print("  %-50s err=%.2e [%s]" % (
        "surrogate sums to ~1",
        err2,
        "OK" if err2 < 1e-6 else "FAIL"))

    probs3 = softmax_surrogate_square(logits, shift=to_qbasis(VDR(1)))
    check("surrogate with shift output[0]", probs3[0])
    print()


def test_attention():
    print("=== attention.py: self_attention ===")
    from vdr.ml.attention import (
        attention_scores, causal_mask, apply_boolean_mask,
        attention_weights, self_attention,
    )

    Q = [vec_to_qbasis(Vec.from_ints([1, 0])),
         vec_to_qbasis(Vec.from_ints([0, 1]))]
    K = [vec_to_qbasis(Vec.from_ints([1, 1])),
         vec_to_qbasis(Vec.from_ints([1, -1]))]
    V = [vec_to_qbasis(Vec.from_ints([10, 0])),
         vec_to_qbasis(Vec.from_ints([0, 10]))]

    scores = attention_scores(Q, K)
    check("attention_scores[0]", scores[0])

    mask = causal_mask(2)
    masked = apply_boolean_mask(scores, mask)
    check("masked_scores[0]", masked[0])

    weights = attention_weights(scores, mask=mask, exp_depth=8)
    check("attention_weights[0]", weights[0])

    out = self_attention(Q, K, V, causal=True, exp_depth=8)
    check_all_vdr("self_attention outputs", out)
    print()


def test_losses():
    print("=== losses.py: mse, l1, mse_grad, l1_grad ===")
    from vdr.ml.losses import mse, l1, mse_grad, l1_grad, hinge_binary

    pred = vec_to_qbasis(Vec.from_ints([1, 2, 3]))
    target = vec_to_qbasis(Vec.from_ints([1, 2, 4]))

    loss = mse(pred, target)
    check("mse loss", loss)

    loss_l1 = l1(pred, target)
    check("l1 loss", loss_l1)

    grad = mse_grad(pred, target)
    check("mse_grad", grad)

    grad_l1 = l1_grad(pred, target)
    check("l1_grad", grad_l1)

    h = hinge_binary(to_qbasis(VDR(1, 2)), 1)
    check("hinge_binary", h)
    print()


def test_optim():
    print("=== optim.py: SGD, Momentum ===")
    from vdr.ml.nn import Linear
    from vdr.ml.optim import SGD, Momentum
    from vdr.ml.losses import mse_grad

    layer = Linear.from_ints([[1, 2], [3, 4]], [0, 0])
    layer.to_qbasis(BITS)

    x = vec_to_qbasis(Vec.from_ints([1, 1]))
    target = vec_to_qbasis(Vec.from_ints([0, 0]))

    opt = SGD(layer.parameters(), lr=VDR(1, 100))

    print("  SGD: 5 training steps")
    for step in range(5):
        opt.zero_grad()
        y = layer.forward(x)
        grad = mse_grad(y, target)
        layer.backward(grad)
        opt.step()
        check("  SGD step %d weight" % (step + 1), layer.weight.value)

    layer2 = Linear.from_ints([[1, 2], [3, 4]], [0, 0])
    layer2.to_qbasis(BITS)

    opt2 = Momentum(layer2.parameters(), lr=VDR(1, 100), beta=VDR(9, 10))

    print("  Momentum: 5 training steps")
    for step in range(5):
        opt2.zero_grad()
        y2 = layer2.forward(x)
        grad2 = mse_grad(y2, target)
        layer2.backward(grad2)
        opt2.step()
        check("  Momentum step %d weight" % (step + 1), layer2.weight.value)
    print()


def test_autodiff():
    print("=== autodiff.py: Node forward+backward ===")
    from vdr.ml.autodiff import Node, ensure_node, relu, mse_loss, sum_nodes, mean_nodes

    a = Node(to_qbasis(VDR(3)))
    b = Node(to_qbasis(VDR(4)))

    c = a * b + a
    check("Node forward (a*b+a)", c.value)

    c.backward()
    check("Node grad a", a.grad)
    check("Node grad b", b.grad)

    d = Node(to_qbasis(VDR(-3)))
    e = relu(d)
    check("relu(-3) value", e.value)
    e.backward()
    check("relu(-3) grad", d.grad)

    pred = [Node(to_qbasis(VDR(1))), Node(to_qbasis(VDR(2)))]
    targets = [to_qbasis(to_qbasis(0)), to_qbasis(VDR(3))]
    loss = mse_loss(pred, targets)
    check("mse_loss value", loss.value)
    loss.backward()
    check("mse_loss grad pred[0]", pred[0].grad)

    nodes = [Node(to_qbasis(VDR(i))) for i in range(4)]
    s = sum_nodes(nodes)
    check("sum_nodes value", s.value)

    m = mean_nodes(nodes)
    check("mean_nodes value", m.value)
    print()


def test_tensor():
    print("=== tensor.py: Tensor3D, batched ops ===")
    from vdr.ml.tensor import (
        Tensor3D, batched_matvec, rowwise_add_bias,
        masked_fill_rows, reduce_sum_rows,
    )

    t = Tensor3D.zero(2, 3, 4)
    check("Tensor3D.zero element", t[0, 0])

    M = mat_to_qbasis(Mat.from_ints([[1, 0], [0, 1]]))
    v = vec_to_qbasis(Vec.from_ints([3, 4]))
    results = batched_matvec([M, M], [v, v])
    check("batched_matvec[0]", results[0])

    rows = [vec_to_qbasis(Vec.from_ints([1, 2])),
            vec_to_qbasis(Vec.from_ints([3, 4]))]
    bias = vec_to_qbasis(Vec.from_ints([10, 10]))
    biased = rowwise_add_bias(rows, bias)
    check("rowwise_add_bias[0]", biased[0])

    mask = [[True, False], [True, True]]
    filled = masked_fill_rows(rows, mask, VDR(-1000))
    check("masked_fill_rows[0]", filled[0])

    total = reduce_sum_rows(rows)
    check("reduce_sum_rows", total)
    print()


def test_transformer():
    print("=== transformer.py: TransformerLM forward ===")
    from vdr.ml.transformer import Embedding, FFNBlock, TransformerBlock, TransformerLM
    from vdr.ml.nn import Linear

    emb = Embedding.from_ints([
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    Wq = Mat.from_ints([[1, 0], [0, 1]])
    Wk = Mat.from_ints([[1, 0], [0, 1]])
    Wv = Mat.from_ints([[1, 0], [0, 1]])
    Wo = Mat.from_ints([[1, 0], [0, 1]])

    ffn = FFNBlock(
        Linear.from_ints([[1, 0], [0, 1]], [0, 0]),
        Linear.from_ints([[1, 0], [0, 1]], [0, 0]),
    )

    block = TransformerBlock(Wq, Wk, Wv, Wo, ffn, causal=True, exp_depth=8)

    output_proj = Linear.from_ints([[1, 0], [0, 1], [1, 1]], [0, 0, 0])

    model = TransformerLM(emb, [block], output_proj)
    model.to_qbasis(BITS)

    check("Wq after to_qbasis", block.Wq)
    check("FFN weight after to_qbasis", ffn.l1.weight.value)
    check("output_proj weight after to_qbasis", output_proj.weight.value)
    check("embedding after to_qbasis", model.embedding.table[0])

    print("  running forward pass...")
    logits = model.forward_logits([0, 1])
    check_all_vdr("forward logits", logits)

    print("  running forward with 3 tokens...")
    logits3 = model.forward_logits([0, 1, 2])
    check_all_vdr("forward logits (3 tokens)", logits3)
    print()


def main():
    setup()
    test_nn()
    test_softmax()
    test_attention()
    test_losses()
    test_optim()
    test_autodiff()
    test_tensor()
    test_transformer()

    print("=" * 60)
    print("passed: %d / %d" % (passed, total))
    if failed:
        print("FAILED: %d" % failed)
    else:
        print("all checks passed")


if __name__ == "__main__":
    main()
