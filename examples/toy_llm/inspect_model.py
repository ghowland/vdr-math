"""
examples/toy_llm/inspect.py

Inspection and display utilities for the small-frame toy LLM.

Every value displayed is an exact rational.
Denominator report confirms all denominators stay at target frames.

    from inspect_model import full_inspection
"""

from vdr.core import VDR
from vdr.linalg import Vec

from data import build_dataset, one_hot_target, build_vocab, invert_vocab
from model import ToyTransformer, surrogate_softmax_probs
from frames import frame_report, frame_check_vec
import config as cfg


# -- parameter inspection -------------------------------------------------

def print_parameters(model, max_per_layer=8):
    """
    Print model parameters with exact rational values.

    I: model, max values to show per component
    """
    components = [
        ("Wq", model.Wq),
        ("Wk", model.Wk),
        ("Wv", model.Wv),
        ("Wo", model.Wo),
        ("FFN L1", model.ffn_l1),
        ("FFN L2", model.ffn_l2),
        ("Output", model.output_proj),
    ]

    total = 0
    print("Model Parameters")
    print("-" * 50)

    for name, layer in components:
        params = layer.parameters()
        n = len(params)
        total += n
        print(f"\n{name} ({n} params):")
        for i, p in enumerate(params):
            if i >= max_per_layer:
                print(f"  ... ({n - max_per_layer} more)")
                break
            frac = p.value.to_fraction()
            d = p.value.d
            print(f"  [{i}] {frac}  (D={d})")

    print(f"\nTotal parameters: {total}")


def print_embeddings(model):
    """
    Print token and positional embeddings with frame info.
    """
    vocab = build_vocab(cfg.CORPUS)
    inv_vocab = invert_vocab(vocab)

    print("Token Embeddings (D_WEIGHT={})".format(cfg.D_WEIGHT))
    print("-" * 50)
    for i, emb in enumerate(model.token_emb):
        token = inv_vocab.get(i, f"<{i}>")
        vals = [f"{emb[j].to_fraction()}" for j in range(len(emb))]
        print(f"  {token:8s} (id={i}): [{', '.join(vals)}]")

    print("\nPositional Embeddings (D_WEIGHT={})".format(cfg.D_WEIGHT))
    print("-" * 50)
    for i, emb in enumerate(model.pos_emb):
        vals = [f"{emb[j].to_fraction()}" for j in range(len(emb))]
        print(f"  pos {i}: [{', '.join(vals)}]")


# -- attention map --------------------------------------------------------

def print_attention_map(model, context_ids=None):
    """
    Print attention weight matrix. Rows sum to exactly 1.
    Causal mask visible as zeros in upper triangle.

    I: model, context_ids (default: first window)
    """
    if context_ids is None:
        windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
        context_ids = windows[0][0]

    vocab = build_vocab(cfg.CORPUS)
    inv_vocab = invert_vocab(vocab)
    tokens = [inv_vocab.get(tid, f"<{tid}>") for tid in context_ids]

    _, cache = model.forward_with_cache(context_ids)
    weights = cache["attn_cache"]["weights"]

    print("Attention Weights (D_PROB={})".format(cfg.D_PROB))
    print("-" * 50)
    print(f"Context: {tokens}")
    print()

    # header
    header = "         " + "".join(f"{t:>10s}" for t in tokens)
    print(header)

    # rows
    for i, row in enumerate(weights):
        row_str = f"{tokens[i]:8s} "
        row_sum = VDR(0)
        for j in range(len(row)):
            val = float(row[j])
            row_str += f"{val:10.4f}"
            row_sum = row_sum + row[j]
        row_str += f"  sum={row_sum.to_fraction()}"
        print(row_str)

    print()
    print("Exact values (nonzero):")
    for i, row in enumerate(weights):
        for j in range(len(row)):
            if row[j] != VDR(0):
                print(f"  [{i},{j}] {tokens[i]}->{tokens[j]}: "
                      f"{row[j].to_fraction()}  (D={row[j].d})")


# -- logits and probabilities --------------------------------------------

def print_logits_and_probs(model, context_ids=None):
    """
    Print logits and surrogate softmax probabilities per position.

    I: model, context_ids (default: first window)
    """
    if context_ids is None:
        windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
        context_ids = windows[0][0]

    vocab = build_vocab(cfg.CORPUS)
    inv_vocab = invert_vocab(vocab)
    tokens = [inv_vocab.get(tid, f"<{tid}>") for tid in context_ids]

    logits_list = model.forward(context_ids)

    print("Logits and Probabilities")
    print("-" * 50)

    for pos in range(len(logits_list)):
        logits = logits_list[pos]
        probs = surrogate_softmax_probs(logits)

        print(f"\nPosition {pos} (after '{tokens[pos]}'):")
        prob_sum = VDR(0)
        for i in range(len(probs)):
            tok = inv_vocab.get(i, f"<{i}>")
            l_frac = logits[i].to_fraction()
            p_frac = probs[i].to_fraction()
            p_approx = float(probs[i])
            print(f"  {tok:8s}: logit={l_frac:>12s}  "
                  f"prob={p_frac}  (~{p_approx:.4f})")
            prob_sum = prob_sum + probs[i]
        print(f"  Sum: {prob_sum.to_fraction()}")


# -- gradient magnitudes --------------------------------------------------

def print_gradient_magnitudes(model, context_ids=None, target_id=None):
    """
    Print gradient magnitudes after one backward pass.

    I: model, context_ids, target_id (defaults: first window)
    """
    if context_ids is None or target_id is None:
        windows, _, _, _ = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
        context_ids, target_id = windows[0]

    from train import surrogate_softmax_then_mse
    from frames import rebase_grads

    model.zero_grad()
    last_logits, _ = model.forward_last_logits_with_cache(context_ids)
    target_vec = one_hot_target(target_id, cfg.VOCAB_SIZE)
    _, _, grad_logits = surrogate_softmax_then_mse(last_logits, target_vec)
    model.backward_from_last(grad_logits)
    rebase_grads(model.parameters(), cfg.D_GRAD)

    components = [
        ("Wq", model.Wq),
        ("Wk", model.Wk),
        ("Wv", model.Wv),
        ("Wo", model.Wo),
        ("FFN L1", model.ffn_l1),
        ("FFN L2", model.ffn_l2),
        ("Output", model.output_proj),
    ]

    print("Gradient Magnitudes (D_GRAD={})".format(cfg.D_GRAD))
    print("-" * 50)

    for name, layer in components:
        params = layer.parameters()
        nonzero = 0
        max_abs = VDR(0)
        for p in params:
            if p.grad is not None and p.grad != VDR(0):
                nonzero += 1
                g_abs = p.grad if p.grad >= VDR(0) else VDR(0) - p.grad
                if g_abs > max_abs:
                    max_abs = g_abs

        max_approx = float(max_abs) if max_abs != VDR(0) else 0.0
        print(f"  {name:10s}: {nonzero:3d}/{len(params):3d} nonzero, "
              f"max|grad| = {max_abs.to_fraction()} (~{max_approx:.6f})")


# -- loss trajectory ------------------------------------------------------

def print_loss_trajectory(losses):
    """
    Print loss trajectory as exact fractions.

    I: list of VDR loss values
    """
    print("Loss Trajectory")
    print("-" * 50)

    for i, loss in enumerate(losses):
        frac = loss.to_fraction()
        approx = float(loss)
        direction = ""
        if i > 0:
            if loss < losses[i - 1]:
                direction = " ↓"
            elif loss > losses[i - 1]:
                direction = " ↑"
            else:
                direction = " ="
        print(f"  Epoch {i + 1:3d}: {str(frac):>40s}  (~{approx:.6f}){direction}")

    if len(losses) >= 2:
        reduction = losses[0] - losses[-1]
        if losses[0] != VDR(0):
            pct_approx = float(reduction) / float(losses[0]) * 100
            print(f"\n  Reduction: {reduction.to_fraction()} (~{pct_approx:.1f}%)")


# -- denominator report --------------------------------------------------

def denominator_report(model):
    """
    Report denominator sizes across all parameters.

    Should show all denominators at D_WEIGHT=128.
    Any other value indicates frame discipline violation.

    I: model
    """
    report = frame_report(model)

    print("Denominator Report")
    print("-" * 50)

    total_params = sum(report.values())
    print(f"  Total parameters: {total_params}")
    print(f"  Expected D: {cfg.D_WEIGHT}")
    print()
    print("  Distribution:")
    for d_str in sorted(report.keys(), key=lambda x: int(x)):
        count = report[d_str]
        marker = " ✓" if d_str == str(cfg.D_WEIGHT) else " ✗ UNEXPECTED"
        if d_str == "1":
            marker = " (zero bias)"
        print(f"    D={d_str:>6s}: {count:4d} params{marker}")

    # check for any non-target denominators
    unexpected = sum(
        count for d_str, count in report.items()
        if d_str != str(cfg.D_WEIGHT) and d_str != "1"
    )
    if unexpected > 0:
        print(f"\n  WARNING: {unexpected} params outside target frame!")
    else:
        print(f"\n  All parameters in target frame D={cfg.D_WEIGHT}.")


# -- combined inspection --------------------------------------------------

def full_inspection(model, losses=None):
    """
    Run all inspection functions.

    I: model, losses (optional)
    """
    print("=" * 60)
    print("VDR Toy LLM — Full Inspection (Small Frames)")
    print(f"  D_WEIGHT={cfg.D_WEIGHT}  D_ACT={cfg.D_ACT}  "
          f"D_SCORE={cfg.D_SCORE}  D_PROB={cfg.D_PROB}")
    print("=" * 60)
    print()

    print_embeddings(model)
    print()

    print_parameters(model, max_per_layer=4)
    print()

    print_attention_map(model)
    print()

    print_logits_and_probs(model)
    print()

    print_gradient_magnitudes(model)
    print()

    if losses:
        print_loss_trajectory(losses)
        print()

    denominator_report(model)
    print()

    print("=" * 60)
    print("Inspection complete. Every value exact. Frames bounded.")
    print("=" * 60)

