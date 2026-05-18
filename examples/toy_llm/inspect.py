"""
Inspection utilities for toy transformer LLM.
Print parameters, attention maps, logits, gradients, denominators.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.basis import to_qbasis, get_default
from vdr.ml.softmax import softmax_surrogate_square

from config import VOCAB_SIZE, SEQ_LEN, CORPUS
from data import build_dataset, one_hot_target, invert_vocab, build_vocab


def print_parameters(model, max_per_layer=4):
    """Print parameter values, limited per layer."""
    print("=== Parameters ===")
    layers = [
        ("Wq.weight", model.Wq.weight),
        ("Wq.bias", model.Wq.bias),
        ("Wk.weight", model.Wk.weight),
        ("Wk.bias", model.Wk.bias),
        ("Wv.weight", model.Wv.weight),
        ("Wv.bias", model.Wv.bias),
        ("Wo.weight", model.Wo.weight),
        ("Wo.bias", model.Wo.bias),
        ("ffn_l1.weight", model.ffn_l1.weight),
        ("ffn_l1.bias", model.ffn_l1.bias),
        ("ffn_l2.weight", model.ffn_l2.weight),
        ("ffn_l2.bias", model.ffn_l2.bias),
        ("output.weight", model.output_proj.weight),
        ("output.bias", model.output_proj.bias),
    ]
    for name, param in layers:
        if hasattr(param.value, "nrows"):
            n = min(max_per_layer, param.value.nrows)
            vals = ["%.4f" % param.value[i, 0].to_float() for i in range(n)]
        else:
            n = min(max_per_layer, len(param.value))
            vals = ["%.4f" % param.value[i].to_float() for i in range(n)]
        print("  %-20s %s%s" % (name, ", ".join(vals),
              "..." if n < (param.value.nrows if hasattr(param.value, "nrows") else len(param.value)) else ""))
    print()


def print_embeddings(model):
    """Print token and positional embeddings."""
    print("=== Token Embeddings ===")
    vocab = build_vocab(CORPUS)
    inv = invert_vocab(vocab)
    for i in range(len(model.token_emb)):
        vals = ["%.3f" % model.token_emb[i][j].to_float()
                for j in range(len(model.token_emb[i]))]
        token = inv.get(i, "?")
        print("  [%d] %-6s %s" % (i, token, ", ".join(vals)))
    print()
    print("=== Positional Embeddings ===")
    for i in range(len(model.pos_emb)):
        vals = ["%.3f" % model.pos_emb[i][j].to_float()
                for j in range(len(model.pos_emb[i]))]
        print("  [%d] %s" % (i, ", ".join(vals)))
    print()


def print_attention_map(model, context_ids):
    """Print attention weight matrix for given context."""
    print("=== Attention Map ===")
    xs = model.embed(context_ids)
    _, cache = model.attention_block_with_cache(xs)

    vocab = build_vocab(CORPUS)
    inv = invert_vocab(vocab)
    tokens = [inv.get(i, "?") for i in context_ids]

    header = "      " + "  ".join("%-6s" % t for t in tokens)
    print(header)

    for i, row in enumerate(cache["weights"]):
        vals = ["%.3f" % row[j].to_float() for j in range(len(row))]
        print("  %-4s %s" % (tokens[i], "  ".join("%-6s" % v for v in vals)))
    print()


def print_logits_and_probs(model, context_ids):
    """Print logits and probabilities for given context."""
    print("=== Logits & Probs ===")
    logits = model.forward_last_logits(context_ids)

    shift = logits[0]
    for k in range(1, len(logits)):
        if logits[k] < shift:
            shift = logits[k]
    probs = softmax_surrogate_square(logits, shift=shift)

    vocab = build_vocab(CORPUS)
    inv = invert_vocab(vocab)

    total = VDR(0)
    for i in range(len(probs)):
        total = total + probs[i]

    for i in range(len(logits)):
        token = inv.get(i, "?")
        print("  %-6s logit=%.4f  prob=%.4f" % (
            token, logits[i].to_float(), probs[i].to_float()))
    print("  sum(probs) = %.10f" % total.to_float())
    print()


def print_gradient_magnitudes(model, context_ids, target_id):
    """Print gradient magnitudes after one forward-backward."""
    from train import surrogate_softmax_then_mse
    print("=== Gradient Magnitudes ===")

    target_vec = one_hot_target(target_id, VOCAB_SIZE)
    model.zero_grad()
    logits, cache = model.forward_with_cache(context_ids)
    _, loss, grad_logits = surrogate_softmax_then_mse(logits[-1], target_vec)
    model.backward_from_last(grad_logits)

    layers = [
        ("Wq.weight", model.Wq.weight),
        ("Wk.weight", model.Wk.weight),
        ("Wv.weight", model.Wv.weight),
        ("Wo.weight", model.Wo.weight),
        ("ffn_l1.weight", model.ffn_l1.weight),
        ("ffn_l2.weight", model.ffn_l2.weight),
        ("output.weight", model.output_proj.weight),
    ]
    for name, param in layers:
        mag = 0.0
        if hasattr(param.grad, "nrows"):
            for i in range(param.grad.nrows):
                for j in range(param.grad.ncols):
                    mag += abs(param.grad[i, j].to_float())
        else:
            for i in range(len(param.grad)):
                mag += abs(param.grad[i].to_float())
        print("  %-20s total_abs_grad = %.6f" % (name, mag))
    print("  loss = %.6f" % loss.to_float())
    print()


def print_loss_trajectory(losses):
    """Print loss per epoch."""
    print("=== Loss Trajectory ===")
    for i, loss in enumerate(losses):
        bar = "#" * max(1, int(loss.to_float() * 50))
        print("  epoch %2d  %.6f  %s" % (i + 1, loss.to_float(), bar))
    print()


def denominator_report(model):
    """
    Report all denominators in model parameters.
    Should all be 2^32.
    """
    print("=== Denominator Report ===")
    denom = 2 ** get_default()
    total = 0
    bad = 0

    for p in model.parameters():
        if hasattr(p, "value"):
            if hasattr(p.value, "nrows"):
                for i in range(p.value.nrows):
                    for j in range(p.value.ncols):
                        total += 1
                        if p.value[i, j].d != denom:
                            bad += 1
            else:
                for i in range(len(p.value)):
                    total += 1
                    if p.value[i].d != denom:
                        bad += 1

    if bad == 0:
        print("  all %d parameters have D = %d  [OK]" % (total, denom))
    else:
        print("  %d / %d parameters have wrong D  [FAIL]" % (bad, total))
    print()


def full_inspection(model, losses=None):
    """Run all inspection functions."""
    windows, vocab, inv_vocab, _ = build_dataset(CORPUS, SEQ_LEN)

    print_parameters(model)
    print_embeddings(model)
    denominator_report(model)

    if windows:
        context_ids, target_id = windows[0]
        print_attention_map(model, context_ids)
        print_logits_and_probs(model, context_ids)
        print_gradient_magnitudes(model, context_ids, target_id)

    if losses:
        print_loss_trajectory(losses)
