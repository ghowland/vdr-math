"""
examples/toy_llm/train.py

Training with fixed D-frames. Rebase grads to D_GRAD after backward,
rebase params to D_WEIGHT after optimizer step.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.ml.optim import SGD

from data import build_dataset, one_hot_target
from model import ToyTransformer, surrogate_softmax, surrogate_softmax_probs
from frames import rebase_grads, rebase_params, zero_vec
from attention_backward import surrogate_softmax_backward
import config as cfg


def mse_loss(pred, target):
    """MSE = (1/n) * sum((pred[i] - target[i])^2)"""
    n = len(pred)
    total = VDR(0)
    for i in range(n):
        diff = pred[i] - target[i]
        total = total + diff * diff
    return total * VDR(1, n)


def mse_loss_grad(pred, target):
    """d(MSE)/d(pred[i]) = (2/n) * (pred[i] - target[i])"""
    n = len(pred)
    scale = VDR(2, n)
    return Vec([(pred[i] - target[i]) * scale for i in range(n)])


def surrogate_softmax_then_mse(logits, target_vec):
    """
    Surrogate softmax + MSE with gradient through both.

    I: logits Vec, target_vec one-hot Vec
    O: (probs, loss, grad_logits)
    """
    probs, shifted = surrogate_softmax(logits)
    loss = mse_loss(probs, target_vec)
    grad_probs = mse_loss_grad(probs, target_vec)
    grad_logits = surrogate_softmax_backward(grad_probs, probs, shifted)
    return probs, loss, grad_logits


def train_step(model, context_ids, target_id, optimizer):
    """
    Single step: forward -> loss -> backward -> rebase grads -> step -> rebase params.

    I: model, context_ids, target_id, optimizer
    O: (loss, probs)
    """
    model.zero_grad()
    last_logits, cache = model.forward_last_logits_with_cache(context_ids)
    target_vec = one_hot_target(target_id, cfg.VOCAB_SIZE)
    probs, loss, grad_logits = surrogate_softmax_then_mse(last_logits, target_vec)
    model.backward_from_last(grad_logits)
    rebase_grads(model.parameters(), cfg.D_GRAD)
    optimizer.step()
    rebase_params(model.parameters(), cfg.D_WEIGHT)
    return loss, probs


def train_epoch(model, windows, optimizer):
    results = []
    for context_ids, target_id in windows:
        loss, probs = train_step(model, context_ids, target_id, optimizer)
        results.append((loss, probs))
    return results


def average_loss(results):
    total = VDR(0)
    for loss, _ in results:
        total = total + loss
    return total * VDR(1, len(results))


def train(n_epochs=None, model=None, verbose=True):
    """Full training loop."""
    if n_epochs is None:
        n_epochs = cfg.N_EPOCHS

    windows, vocab, inv_vocab, vocab_size = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
    assert vocab_size == cfg.VOCAB_SIZE

    if model is None:
        model = ToyTransformer()

    optimizer = SGD(model.parameters(), lr=cfg.LR)
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        results = train_epoch(model, windows, optimizer)
        avg_loss = average_loss(results)
        loss_history.append(avg_loss)

        if verbose:
            frac = avg_loss.to_fraction()
            approx = float(avg_loss)
            print(f"Epoch {epoch:3d}: loss = {frac}  (~{approx:.6f})")

            for i, (_, probs_i) in enumerate(results):
                prob_sum = VDR(0)
                for j in range(len(probs_i)):
                    prob_sum = prob_sum + probs_i[j]
                assert prob_sum == VDR(1), (
                    f"Epoch {epoch} window {i}: softmax sum = "
                    f"{prob_sum.to_fraction()}"
                )

    if verbose:
        print(f"\nTraining complete. {n_epochs} epochs, "
              f"{len(windows)} windows/epoch.")
        print(f"Final loss: {loss_history[-1].to_fraction()}")

    return model, loss_history


def quick_train(n_epochs=5):
    return train(n_epochs=n_epochs, verbose=True)


def train_silent(n_epochs=None, model=None):
    return train(n_epochs=n_epochs, model=model, verbose=False)
