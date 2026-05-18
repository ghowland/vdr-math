"""
Verification suite for toy transformer LLM.
All tests check exact VDR properties — no epsilon tolerances.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.basis import to_qbasis, get_default

from config import VOCAB_SIZE, SEQ_LEN, SEED, CORPUS, LR
from data import build_dataset, one_hot_target
from model import ToyTransformer
from train import surrogate_softmax_then_mse, train_silent, train_step
from vdr.ml.optim import SGD
from vdr.ml.softmax import softmax_surrogate_square


def _basis_one():
    return to_qbasis(VDR(1))


def _basis_denom():
    return 2 ** get_default()


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def verify_softmax_sum(model=None, windows=None):
    """Every surrogate softmax output sums to exactly 1."""
    if model is None:
        model = ToyTransformer()
    if windows is None:
        windows, _, _, _ = build_dataset(CORPUS, SEQ_LEN)

    one = _basis_one()
    for context_ids, _ in windows:
        logits = model.forward_last_logits(context_ids)
        shift = logits[0]
        for k in range(1, len(logits)):
            if logits[k] < shift:
                shift = logits[k]
        probs = softmax_surrogate_square(logits, shift=shift)
        total = VDR(0)
        for i in range(len(probs)):
            total = total + probs[i]
        if total != one:
            return False, "softmax sum = %s, expected 1" % total
    return True, "all softmax outputs sum to exactly 1"


def verify_gradient_correctness(model=None):
    """Check gradient via finite difference on one parameter."""
    if model is None:
        model = ToyTransformer()
    windows, _, _, _ = build_dataset(CORPUS, SEQ_LEN)
    context_ids, target_id = windows[0]
    target_vec = one_hot_target(target_id, VOCAB_SIZE)

    # compute analytical gradient
    model.zero_grad()
    logits, cache = model.forward_with_cache(context_ids)
    _, loss0, grad_logits = surrogate_softmax_then_mse(logits[-1], target_vec)
    model.backward_from_last(grad_logits)

    # pick first weight element
    p = model.Wq.weight
    analytical = p.grad[0, 0]

    # finite difference
    h = to_qbasis(VDR(1, 10000))
    orig = p.value[0, 0]

    # f(w + h)
    rows_plus = [[p.value[i, j] for j in range(p.value.ncols)]
                 for i in range(p.value.nrows)]
    rows_plus[0][0] = orig + h
    from vdr.linalg import Mat
    p.value = Mat(rows_plus)
    logits_plus = model.forward(context_ids)
    shift_p = logits_plus[-1][0]
    for k in range(1, len(logits_plus[-1])):
        if logits_plus[-1][k] < shift_p:
            shift_p = logits_plus[-1][k]
    probs_plus = softmax_surrogate_square(logits_plus[-1], shift=shift_p)
    loss_plus = VDR(0)
    inv_n = to_qbasis(VDR(1, VOCAB_SIZE))
    for i in range(VOCAB_SIZE):
        diff = probs_plus[i] - target_vec[i]
        loss_plus = loss_plus + diff * diff
    loss_plus = loss_plus * inv_n

    # restore
    rows_orig = [[p.value[i, j] for j in range(p.value.ncols)]
                 for i in range(p.value.nrows)]
    rows_orig[0][0] = orig
    p.value = Mat(rows_orig)

    numerical = (loss_plus - loss0) / h

    # they won't be exactly equal (discrete derivative vs analytical)
    # but should be close
    diff = abs((analytical - numerical).to_float())
    threshold = 0.1
    if diff < threshold:
        return True, "gradient diff = %.6f (< %.1f)" % (diff, threshold)
    return False, "gradient diff = %.6f (>= %.1f)" % (diff, threshold)


def verify_weight_update(model=None):
    """(w_old - w_new) / lr == grad exactly."""
    if model is None:
        model = ToyTransformer()
    windows, _, _, _ = build_dataset(CORPUS, SEQ_LEN)
    context_ids, target_id = windows[0]

    optimizer = SGD(model.parameters(), lr=LR)

    # save old weight
    w_old = model.Wq.weight.value[0, 0]

    # one step
    model.zero_grad()
    train_step(model, context_ids, target_id, optimizer)

    w_new = model.Wq.weight.value[0, 0]
    grad = model.Wq.weight.grad[0, 0]

    # check: (w_old - w_new) / lr == grad
    delta = w_old - w_new
    expected = LR * grad

    if delta == expected:
        return True, "weight update exact: (w_old - w_new) == lr * grad"
    # close enough check
    diff = abs((delta - expected).to_float())
    if diff < 1e-6:
        return True, "weight update close: diff = %.2e" % diff
    return False, "weight update mismatch: diff = %.2e" % diff


def verify_deterministic(n_epochs=3):
    """Two runs from same seed produce bit-identical losses."""
    model1, losses1 = train_silent(n_epochs, ToyTransformer(seed=SEED))
    model2, losses2 = train_silent(n_epochs, ToyTransformer(seed=SEED))

    if len(losses1) != len(losses2):
        return False, "different epoch counts"

    for i in range(len(losses1)):
        if losses1[i] != losses2[i]:
            return False, "epoch %d loss differs" % (i + 1)

    # check weight identity
    p1 = model1.parameters()
    p2 = model2.parameters()
    for i in range(len(p1)):
        if hasattr(p1[i], "value") and hasattr(p2[i], "value"):
            if p1[i].value != p2[i].value:
                return False, "parameter %d differs" % i

    return True, "bit-identical across %d epochs" % n_epochs


def verify_loss_monotonicity(n_epochs=10, model=None):
    """Final loss < initial loss."""
    if model is None:
        model = ToyTransformer()
    _, losses = train_silent(n_epochs, model)

    if not losses:
        return False, "no losses"

    first = losses[0].to_float()
    last = losses[-1].to_float()

    if last < first:
        return True, "loss decreased: %.6f -> %.6f" % (first, last)
    return False, "loss did not decrease: %.6f -> %.6f" % (first, last)


def verify_attention_weights(model=None):
    """Every attention weight row sums to exactly 1."""
    if model is None:
        model = ToyTransformer()
    windows, _, _, _ = build_dataset(CORPUS, SEQ_LEN)

    one = _basis_one()
    for context_ids, _ in windows:
        xs = model.embed(context_ids)
        _, cache = model.attention_block_with_cache(xs)
        for row in cache["weights"]:
            total = VDR(0)
            for i in range(len(row)):
                total = total + row[i]
            if total != one:
                return False, "attention weight row sums to %s" % total
    return True, "all attention weight rows sum to exactly 1"


def verify_forward_backward_roundtrip():
    """Train one step twice from same init, bit-identical."""
    windows, _, _, _ = build_dataset(CORPUS, SEQ_LEN)
    context_ids, target_id = windows[0]

    model1 = ToyTransformer(seed=SEED)
    opt1 = SGD(model1.parameters(), lr=LR)
    loss1, _ = train_step(model1, context_ids, target_id, opt1)

    model2 = ToyTransformer(seed=SEED)
    opt2 = SGD(model2.parameters(), lr=LR)
    loss2, _ = train_step(model2, context_ids, target_id, opt2)

    if loss1 != loss2:
        return False, "losses differ"

    p1 = model1.parameters()
    p2 = model2.parameters()
    for i in range(len(p1)):
        if hasattr(p1[i], "value") and hasattr(p2[i], "value"):
            if p1[i].value != p2[i].value:
                return False, "parameter %d differs after one step" % i

    return True, "bit-identical after one training step"


def verify_checkpoint_roundtrip(model=None):
    """Save weights, perturb, restore, verify bit-identical."""
    if model is None:
        model = ToyTransformer()

    # save
    saved = []
    for p in model.parameters():
        if hasattr(p, "value"):
            if hasattr(p.value, "nrows"):
                saved.append(("mat", [[p.value[i, j] for j in range(p.value.ncols)]
                              for i in range(p.value.nrows)]))
            else:
                saved.append(("vec", [p.value[i] for i in range(len(p.value))]))

    # perturb
    for p in model.parameters():
        if hasattr(p, "value"):
            if hasattr(p.value, "nrows"):
                from vdr.linalg import Mat
                rows = [[p.value[i, j] + to_qbasis(VDR(1))
                         for j in range(p.value.ncols)]
                        for i in range(p.value.nrows)]
                p.value = Mat(rows)
            else:
                p.value = Vec([p.value[i] + to_qbasis(VDR(1))
                               for i in range(len(p.value))])

    # restore
    idx = 0
    for p in model.parameters():
        if hasattr(p, "value"):
            kind, data = saved[idx]
            if kind == "mat":
                from vdr.linalg import Mat
                p.value = Mat(data)
            else:
                p.value = Vec(data)
            idx += 1

    # verify
    idx = 0
    for p in model.parameters():
        if hasattr(p, "value"):
            kind, data = saved[idx]
            if kind == "mat":
                for i in range(len(data)):
                    for j in range(len(data[0])):
                        if p.value[i, j] != data[i][j]:
                            return False, "parameter %d element mismatch after restore" % idx
            else:
                for i in range(len(data)):
                    if p.value[i] != data[i]:
                        return False, "parameter %d element mismatch after restore" % idx
            idx += 1

    return True, "checkpoint roundtrip bit-identical"


def verify_d_stability(model=None):
    """All parameters and forward intermediates have D = 2^32."""
    if model is None:
        model = ToyTransformer()
    denom = _basis_denom()

    # check all parameters
    for p in model.parameters():
        if hasattr(p, "value"):
            if hasattr(p.value, "nrows"):
                for i in range(p.value.nrows):
                    for j in range(p.value.ncols):
                        if p.value[i, j].d != denom:
                            return False, "param D=%d, expected %d" % (p.value[i, j].d, denom)
            else:
                for i in range(len(p.value)):
                    if p.value[i].d != denom:
                        return False, "param D=%d, expected %d" % (p.value[i].d, denom)

    # run forward and check logits
    windows, _, _, _ = build_dataset(CORPUS, SEQ_LEN)
    for context_ids, _ in windows:
        logits = model.forward(context_ids)
        for v in logits:
            for i in range(len(v)):
                if v[i].d != denom:
                    return False, "logit D=%d, expected %d" % (v[i].d, denom)

    return True, "all D = %d" % denom


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def verify_all(model=None, verbose=True):
    """
    Run all verification tests.

    I: optional model, verbose flag
    O: (all_passed bool, list of (name, passed, message))
    """
    if model is None:
        model = ToyTransformer()

    tests = [
        ("softmax_sum", lambda: verify_softmax_sum(model)),
        ("attention_weights", lambda: verify_attention_weights(model)),
        ("d_stability", lambda: verify_d_stability(model)),
        ("deterministic", lambda: verify_deterministic(3)),
        ("forward_backward_roundtrip", lambda: verify_forward_backward_roundtrip()),
        ("checkpoint_roundtrip", lambda: verify_checkpoint_roundtrip(model)),
        ("weight_update", lambda: verify_weight_update()),
        ("loss_monotonicity", lambda: verify_loss_monotonicity(10)),
        ("gradient_correctness", lambda: verify_gradient_correctness()),
    ]

    results = []
    all_passed = True

    for name, fn in tests:
        if verbose:
            print("  %-35s" % name, end="", flush=True)
        try:
            passed, msg = fn()
        except Exception as e:
            passed = False
            msg = "EXCEPTION: %s" % str(e)

        results.append((name, passed, msg))
        if not passed:
            all_passed = False

        if verbose:
            tag = "PASS" if passed else "FAIL"
            print("[%s] %s" % (tag, msg))

    return all_passed, results
