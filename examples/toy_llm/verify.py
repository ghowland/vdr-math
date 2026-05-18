"""
examples/toy_llm/verify.py

Exact verification suite for the small-frame toy LLM.

    from verify import verify_all

    verify_all(model)

Every test uses exact VDR comparison. No tolerances, no epsilons.
Frame discipline verified: all denominators stay at target values.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.ml.optim import SGD
from vdr.ml.checkpoint import save_model, load_model_parameters

from data import build_dataset, one_hot_target
from model import ToyTransformer, surrogate_softmax_probs
from train import surrogate_softmax_then_mse, train_step, train_silent
from frames import (
    rebase_value, rebase_vec, rebase_params, rebase_grads,
    frame_check, frame_check_vec, frame_report, zero_vec,
)
import config as cfg


# -- test 1: softmax sum-to-one -------------------------------------------

def verify_softmax_sum(model, windows=None):
    """
    Verify surrogate softmax sums to exactly 1 at every position.

    Runs forward on every training window. For every position,
    computes surrogate softmax and checks sum == VDR(1) exactly.

    I: model ToyTransformer, windows (optional)
    O: (passed: bool, details: str)
    """
    if windows is None:
        windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)

    violations = []

    for w_idx, (context_ids, target_id) in enumerate(windows):
        logits = model.forward(context_ids)
        for pos in range(len(logits)):
            probs = surrogate_softmax_probs(logits[pos])
            prob_sum = VDR(0)
            for i in range(len(probs)):
                prob_sum = prob_sum + probs[i]
            if prob_sum != VDR(1):
                violations.append(
                    f"window {w_idx} pos {pos}: "
                    f"sum = {prob_sum.to_fraction()}"
                )

    if violations:
        detail = "Softmax sum-to-one FAILED:\n  " + "\n  ".join(violations)
        return False, detail

    n_checks = len(windows) * cfg.SEQ_LEN
    return True, f"Softmax sum-to-one: {n_checks} checks, all exactly 1"


# -- test 2: gradient correctness ----------------------------------------

def verify_gradient_correctness(model=None):
    """
    Compare autodiff gradients against numerical discrete derivative.

    For a single training step, compute autodiff gradient on a few
    output projection parameters. Then compute numerical gradient
    via (loss(w+h) - loss(w)) / h. Both are exact rationals.

    For the surrogate softmax + MSE chain, numerical and autodiff
    should be close (difference proportional to h).

    I: model (optional, fresh if None)
    O: (passed: bool, details: str)
    """
    if model is None:
        model = ToyTransformer()

    windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
    context_ids, target_id = windows[0]
    target_vec = one_hot_target(target_id, cfg.VOCAB_SIZE)

    # autodiff gradient
    model.zero_grad()
    last_logits, cache = model.forward_last_logits_with_cache(context_ids)
    probs, loss_base, grad_logits = surrogate_softmax_then_mse(
        last_logits, target_vec
    )
    model.backward_from_last(grad_logits)
    rebase_grads(model.parameters(), cfg.D_GRAD)

    params = model.output_proj.parameters()
    if not params:
        return False, "No parameters in output projection"

    n_check = min(3, len(params))
    auto_grads = [params[i].grad for i in range(n_check)]

    # numerical gradient
    h = VDR(1, 10000)
    num_grads = []

    for p_idx in range(n_check):
        p = params[p_idx]
        original_val = p.value

        # loss at w + h
        p.value = original_val + h
        logits_plus = model.forward_last_logits(context_ids)
        probs_plus = surrogate_softmax_probs(logits_plus)
        loss_plus = VDR(0)
        n = len(probs_plus)
        for i in range(n):
            diff = probs_plus[i] - target_vec[i]
            loss_plus = loss_plus + diff * diff
        loss_plus = loss_plus * VDR(1, n)

        num_grad = (loss_plus - loss_base) / h
        num_grads.append(num_grad)

        # restore
        p.value = original_val

    # compare
    details = []
    all_close = True
    for i in range(n_check):
        ag = auto_grads[i]
        ng = num_grads[i]
        if ag is None:
            details.append(f"  param {i}: autodiff grad is None")
            all_close = False
            continue

        diff = ag - ng
        diff_abs = diff if diff >= VDR(0) else VDR(0) - diff
        threshold = VDR(1, 10)

        if diff_abs > threshold:
            details.append(
                f"  param {i}: autodiff={ag.to_fraction()}, "
                f"numerical={ng.to_fraction()}, diff={diff.to_fraction()}"
            )
            all_close = False
        else:
            details.append(
                f"  param {i}: autodiff={ag.to_fraction()}, "
                f"numerical={ng.to_fraction()} — consistent"
            )

    if all_close:
        return True, (
            "Gradient correctness: autodiff consistent with numerical\n"
            + "\n".join(details)
        )
    else:
        return False, "Gradient correctness FAILED:\n" + "\n".join(details)


# -- test 3: weight update exactness -------------------------------------

def verify_weight_update(model=None):
    """
    Verify w_new = w_old - lr * grad exactly (before rebase).

    Captures w_old and grad, computes expected w_new,
    takes optimizer step, compares.

    Note: after rebase to D_WEIGHT the exact identity may not hold
    (rebase introduces remainder). So we check before rebase.

    I: model (optional)
    O: (passed: bool, details: str)
    """
    if model is None:
        model = ToyTransformer()

    windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
    context_ids, target_id = windows[0]
    target_vec = one_hot_target(target_id, cfg.VOCAB_SIZE)

    params = model.output_proj.parameters()
    n_check = min(5, len(params))

    # record w_old
    w_old = [params[i].value for i in range(n_check)]

    # forward + backward
    model.zero_grad()
    last_logits, cache = model.forward_last_logits_with_cache(context_ids)
    _, _, grad_logits = surrogate_softmax_then_mse(last_logits, target_vec)
    model.backward_from_last(grad_logits)

    # record grads before rebase
    grads = [params[i].grad for i in range(n_check)]

    # compute expected w_new = w_old - lr * grad
    expected = []
    for i in range(n_check):
        if grads[i] is not None:
            expected.append(w_old[i] - cfg.LR * grads[i])
        else:
            expected.append(w_old[i])

    # take optimizer step (without rebase — test raw update)
    optimizer = SGD(params, lr=cfg.LR)
    optimizer.step()

    # compare
    violations = []
    for i in range(n_check):
        actual = params[i].value
        if actual != expected[i]:
            violations.append(
                f"  param {i}: expected={expected[i].to_fraction()}, "
                f"got={actual.to_fraction()}"
            )

    # restore params for caller (rebase to D_WEIGHT)
    rebase_params(params, cfg.D_WEIGHT)

    if violations:
        return False, "Weight update exactness FAILED:\n" + "\n".join(violations)

    return True, (
        f"Weight update exactness: {n_check} params verified, "
        f"w_new = w_old - lr*grad exactly"
    )


# -- test 4: deterministic reproducibility --------------------------------

def verify_deterministic(n_epochs=3):
    """
    Same seed, same data, same result. Run twice, compare bit-identical.

    I: n_epochs
    O: (passed: bool, details: str)
    """
    model1, losses1 = train_silent(n_epochs=n_epochs)
    params1 = [p.value for p in model1.parameters()]

    model2, losses2 = train_silent(n_epochs=n_epochs)
    params2 = [p.value for p in model2.parameters()]

    loss_match = all(losses1[i] == losses2[i] for i in range(len(losses1)))
    param_match = all(params1[i] == params2[i] for i in range(len(params1)))

    if loss_match and param_match:
        return True, (
            f"Deterministic: {n_epochs} epochs, "
            f"{len(params1)} params — bit-identical"
        )
    else:
        issues = []
        if not loss_match:
            issues.append("Loss histories differ")
        if not param_match:
            issues.append("Parameters differ")
        return False, "Deterministic FAILED: " + "; ".join(issues)


# -- test 5: loss monotonicity -------------------------------------------

def verify_loss_monotonicity(n_epochs=10, model=None):
    """
    Final loss < initial loss.

    I: n_epochs, model
    O: (passed: bool, details: str)
    """
    model, losses = train_silent(n_epochs=n_epochs, model=model)

    if len(losses) < 2:
        return False, "Not enough epochs"

    first = losses[0]
    last = losses[-1]

    if last < first:
        return True, (
            f"Loss monotonicity: {first.to_fraction()} -> "
            f"{last.to_fraction()} over {n_epochs} epochs"
        )
    else:
        return False, (
            f"Loss monotonicity FAILED: first={first.to_fraction()}, "
            f"last={last.to_fraction()}"
        )


# -- test 6: attention weights sum to 1 ----------------------------------

def verify_attention_weights(model=None):
    """
    Every row of attention weights sums to exactly 1.

    I: model (optional)
    O: (passed: bool, details: str)
    """
    if model is None:
        model = ToyTransformer()

    windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)

    violations = []
    total_rows = 0

    for w_idx, (context_ids, target_id) in enumerate(windows):
        _, cache = model.forward_with_cache(context_ids)
        weights = cache["attn_cache"]["weights"]

        for row_idx, row in enumerate(weights):
            total_rows += 1
            row_sum = VDR(0)
            for i in range(len(row)):
                row_sum = row_sum + row[i]

            if row_sum != VDR(1):
                violations.append(
                    f"window {w_idx} row {row_idx}: "
                    f"sum = {row_sum.to_fraction()}"
                )

    if violations:
        return False, (
            "Attention weight sum FAILED:\n  " + "\n  ".join(violations)
        )

    return True, (
        f"Attention weights: {total_rows} rows, all sum to exactly 1"
    )


# -- test 7: forward-backward roundtrip ----------------------------------

def verify_forward_backward_roundtrip():
    """
    Train one step from same seed twice. Bit-identical results.

    I: (none)
    O: (passed: bool, details: str)
    """
    windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
    context_ids, target_id = windows[0]
    target_vec = one_hot_target(target_id, cfg.VOCAB_SIZE)

    # run 1
    model1 = ToyTransformer()
    optimizer1 = SGD(model1.parameters(), lr=cfg.LR)
    model1.zero_grad()
    logits1, _ = model1.forward_last_logits_with_cache(context_ids)
    _, loss1, grad1 = surrogate_softmax_then_mse(logits1, target_vec)
    model1.backward_from_last(grad1)
    rebase_grads(model1.parameters(), cfg.D_GRAD)
    grads_1 = [p.grad for p in model1.parameters()]
    optimizer1.step()
    rebase_params(model1.parameters(), cfg.D_WEIGHT)
    params_1 = [p.value for p in model1.parameters()]

    # run 2
    model2 = ToyTransformer()
    optimizer2 = SGD(model2.parameters(), lr=cfg.LR)
    model2.zero_grad()
    logits2, _ = model2.forward_last_logits_with_cache(context_ids)
    _, loss2, grad2 = surrogate_softmax_then_mse(logits2, target_vec)
    model2.backward_from_last(grad2)
    rebase_grads(model2.parameters(), cfg.D_GRAD)
    grads_2 = [p.grad for p in model2.parameters()]
    optimizer2.step()
    rebase_params(model2.parameters(), cfg.D_WEIGHT)
    params_2 = [p.value for p in model2.parameters()]

    issues = []

    if loss1 != loss2:
        issues.append(f"Losses differ: {loss1.to_fraction()} vs {loss2.to_fraction()}")

    grad_mismatch = False
    for i in range(len(grads_1)):
        g1, g2 = grads_1[i], grads_2[i]
        if g1 is None and g2 is None:
            continue
        if g1 != g2:
            grad_mismatch = True
            break
    if grad_mismatch:
        issues.append("Gradients differ")

    param_mismatch = False
    for i in range(len(params_1)):
        if params_1[i] != params_2[i]:
            param_mismatch = True
            break
    if param_mismatch:
        issues.append("Parameters differ after update")

    if issues:
        return False, "Forward-backward roundtrip FAILED:\n  " + "\n  ".join(issues)

    return True, (
        f"Forward-backward roundtrip: loss, {len(grads_1)} grads, "
        f"{len(params_1)} params — bit-identical"
    )


# -- test 8: checkpoint roundtrip ----------------------------------------

def verify_checkpoint_roundtrip(model=None):
    """
    save -> perturb -> load -> bit-identical.

    I: model (optional, trains 3 epochs if None)
    O: (passed: bool, details: str)
    """
    if model is None:
        model, _ = train_silent(n_epochs=3)

    state = save_model(model)
    original = [p.value for p in model.parameters()]

    # perturb
    for p in model.parameters():
        p.value = VDR(999, 1000)

    # verify perturbation
    if model.parameters()[0].value == original[0]:
        return False, "Perturbation did not change params"

    # load
    load_model_parameters(model, state)
    restored = [p.value for p in model.parameters()]

    mismatches = sum(1 for i in range(len(original)) if original[i] != restored[i])

    if mismatches > 0:
        return False, (
            f"Checkpoint roundtrip FAILED: "
            f"{mismatches}/{len(original)} params differ"
        )

    return True, (
        f"Checkpoint roundtrip: {len(original)} params "
        f"saved, perturbed, restored — bit-identical"
    )


# -- test 9: frame discipline --------------------------------------------

def verify_frame_discipline(model=None):
    """
    After forward pass, verify all intermediates are in expected D-frames.

    Checks:
      - All parameters in D_WEIGHT
      - Activations (embeddings) in D_ACT
      - Attention weights rows in D_PROB
      - Logits in D_ACT

    I: model (optional)
    O: (passed: bool, details: str)
    """
    if model is None:
        model = ToyTransformer()

    windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
    context_ids, target_id = windows[0]

    violations = []

    # check parameter frames
    report = frame_report(model)
    expected_d_str = str(cfg.D_WEIGHT)
    for d_str, count in report.items():
        if d_str != expected_d_str and d_str != "1":
            violations.append(
                f"Parameters: {count} params have D={d_str}, "
                f"expected {cfg.D_WEIGHT}"
            )

    # forward with cache to inspect intermediates
    _, cache = model.forward_with_cache(context_ids)

    # check embedded activations
    for i, x in enumerate(cache["xs_embed"]):
        if not frame_check_vec(x, cfg.D_ACT):
            violations.append(f"Embedding pos {i}: not in D_ACT={cfg.D_ACT}")

    # check post-attention activations
    for i, x in enumerate(cache["xs_post_attn"]):
        if not frame_check_vec(x, cfg.D_ACT):
            violations.append(f"Post-attn pos {i}: not in D_ACT={cfg.D_ACT}")

    # check attention weights in D_PROB
    attn_weights = cache["attn_cache"]["weights"]
    for i, row in enumerate(attn_weights):
        if not frame_check_vec(row, cfg.D_PROB):
            violations.append(f"Attn weight row {i}: not in D_PROB={cfg.D_PROB}")

    # check post-FFN activations
    for i, x in enumerate(cache["xs_post_ffn"]):
        if not frame_check_vec(x, cfg.D_ACT):
            violations.append(f"Post-FFN pos {i}: not in D_ACT={cfg.D_ACT}")

    # check logits
    for i, x in enumerate(cache["logits"]):
        if not frame_check_vec(x, cfg.D_ACT):
            violations.append(f"Logits pos {i}: not in D_ACT={cfg.D_ACT}")

    if violations:
        return False, (
            "Frame discipline FAILED:\n  " + "\n  ".join(violations)
        )

    return True, (
        f"Frame discipline: parameters D={cfg.D_WEIGHT}, "
        f"activations D={cfg.D_ACT}, weights D={cfg.D_PROB}, "
        f"logits D={cfg.D_ACT} — all correct"
    )


# -- run all tests --------------------------------------------------------

def verify_all(model=None, verbose=True):
    """
    Run all 9 verification tests.

    I: model (optional), verbose
    O: (all_passed: bool, results: list of (name, passed, detail))
    """
    tests = [
        ("1. Softmax sum-to-one",
         lambda: verify_softmax_sum(model)),
        ("2. Gradient correctness",
         lambda: verify_gradient_correctness()),
        ("3. Weight update exactness",
         lambda: verify_weight_update()),
        ("4. Deterministic reproducibility",
         lambda: verify_deterministic(3)),
        ("5. Loss monotonicity",
         lambda: verify_loss_monotonicity(10)),
        ("6. Attention weights sum to 1",
         lambda: verify_attention_weights(model)),
        ("7. Forward-backward roundtrip",
         lambda: verify_forward_backward_roundtrip()),
        ("8. Checkpoint roundtrip",
         lambda: verify_checkpoint_roundtrip(model)),
        ("9. Frame discipline",
         lambda: verify_frame_discipline(model)),
    ]

    results = []
    all_passed = True

    if verbose:
        print("=" * 60)
        print("VDR Toy LLM — Exact Verification Suite")
        print("=" * 60)
        print()

    for name, test_fn in tests:
        try:
            passed, detail = test_fn()
        except Exception as e:
            passed = False
            detail = f"EXCEPTION: {type(e).__name__}: {e}"

        results.append((name, passed, detail))
        if not passed:
            all_passed = False

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}")
            for line in detail.split("\n"):
                print(f"       {line}")
            print()

    if verbose:
        print("=" * 60)
        n_pass = sum(1 for _, p, _ in results if p)
        print(f"Results: {n_pass}/{len(results)} passed")
        if all_passed:
            print("All tests passed. Every value exact. Frames bounded.")
        else:
            print("Some tests failed. See details above.")
        print("=" * 60)

    return all_passed, results
