"""
examples/toy_llm/train.py

Toy LLM training with small fixed D-frames.

Rebase discipline: after every backward pass, gradients rebased to D_GRAD.
After every optimizer step, parameters rebased to D_WEIGHT.
Denominators never grow. Arithmetic stays fast.

    from train import train

    model, losses = train(n_epochs=20)
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.ml.optim import SGD

from data import build_dataset, one_hot_target
from model import ToyTransformer, surrogate_softmax, surrogate_softmax_probs
from frames import rebase_vec, rebase_grads, rebase_params, zero_vec
from attention_backward import surrogate_softmax_backward
import config as cfg


# -- loss computation -----------------------------------------------------

def mse_loss(pred, target):
    """
    Mean squared error between two Vec.

    MSE = (1/n) * sum((pred[i] - target[i])^2)

    I: pred Vec, target Vec (same length)
    O: VDR scalar — exact rational

        mse_loss(Vec([VDR(1,2), VDR(1)]), Vec([VDR(0), VDR(1)]))
        -> VDR(1, 8)
    """
    n = len(pred)
    total = VDR(0)
    for i in range(n):
        diff = pred[i] - target[i]
        total = total + diff * diff
    return total * VDR(1, n)


def mse_loss_grad(pred, target):
    """
    Gradient of MSE w.r.t. pred.

    d(MSE)/d(pred[i]) = (2/n) * (pred[i] - target[i])

    I: pred Vec, target Vec
    O: Vec — exact gradient
    """
    n = len(pred)
    scale = VDR(2, n)
    result = []
    for i in range(n):
        result.append((pred[i] - target[i]) * scale)
    return Vec(result)


def surrogate_softmax_then_mse(logits, target_vec):
    """
    Surrogate softmax + MSE loss, with gradient through both.

    I: logits Vec in D_ACT, target_vec Vec (one-hot)
    O: (probs, loss, grad_logits)
       probs: Vec summing to exactly 1
       loss: VDR scalar
       grad_logits: Vec — gradient of loss w.r.t. logits

    Uses surrogate softmax: s_i = (z_i-m+c)^2 / sum((z_j-m+c)^2)
    No transcendentals. Exact sum-to-one.
    """
    # surrogate softmax
    probs, shifted = surrogate_softmax(logits)

    # MSE loss
    loss = mse_loss(probs, target_vec)

    # gradient of MSE w.r.t. probs
    grad_probs = mse_loss_grad(probs, target_vec)

    # gradient through surrogate softmax
    grad_logits = surrogate_softmax_backward(grad_probs, probs, shifted)

    return probs, loss, grad_logits


# -- training step --------------------------------------------------------

def train_step(model, context_ids, target_id, optimizer):
    """
    Single training step with frame discipline.

    Forward -> loss -> backward -> rebase grads to D_GRAD
    -> optimizer step -> rebase params to D_WEIGHT.

    I: model ToyTransformer, context_ids list[int],
       target_id int, optimizer SGD
    O: (loss VDR, probs Vec)
    S: model parameters updated in place, all in D_WEIGHT frame

        loss, probs = train_step(model, [0,1,2,3], 0, optimizer)
    """
    # zero gradients
    model.zero_grad()

    # forward with cache
    last_logits, cache = model.forward_last_logits_with_cache(context_ids)

    # target one-hot
    target_vec = one_hot_target(target_id, cfg.VOCAB_SIZE)

    # softmax + loss + gradient
    probs, loss, grad_logits = surrogate_softmax_then_mse(last_logits, target_vec)

    # backward through model
    model.backward_from_last(grad_logits)

    # rebase gradients to D_GRAD before optimizer step
    rebase_grads(model.parameters(), cfg.D_GRAD)

    # optimizer step
    optimizer.step()

    # rebase parameters back to D_WEIGHT
    rebase_params(model.parameters(), cfg.D_WEIGHT)

    return loss, probs


def train_epoch(model, windows, optimizer):
    """
    Train one epoch over all windows.

    I: model, list of (context_ids, target_id), optimizer
    O: list of (loss, probs) per window — all exact
    """
    results = []
    for context_ids, target_id in windows:
        loss, probs = train_step(model, context_ids, target_id, optimizer)
        results.append((loss, probs))
    return results


def average_loss(results):
    """
    Compute average loss from epoch results.

    I: list of (loss, probs) tuples
    O: VDR — exact average
    """
    total = VDR(0)
    for loss, _ in results:
        total = total + loss
    return total * VDR(1, len(results))


# -- full training loop ---------------------------------------------------

def train(n_epochs=None, model=None, verbose=True):
    """
    Full training loop with frame discipline.

    I: n_epochs (default from config), model (default: fresh),
       verbose (print losses)
    O: (model, loss_history)
       loss_history: list of VDR — average loss per epoch, all exact

    Every loss is an exact fraction.
    Every softmax sums to exactly 1.
    Every gradient is exact.
    Parameters stay in D_WEIGHT=128 frame throughout.

        model, losses = train()
    """
    if n_epochs is None:
        n_epochs = cfg.N_EPOCHS

    # build dataset
    windows, vocab, inv_vocab, vocab_size = build_dataset(cfg.CORPUS, cfg.SEQ_LEN)
    assert vocab_size == cfg.VOCAB_SIZE, (
        f"Corpus has {vocab_size} tokens, config says {cfg.VOCAB_SIZE}"
    )

    # build model
    if model is None:
        model = ToyTransformer()

    # optimizer
    optimizer = SGD(model.parameters(), lr=cfg.LR)

    # training loop
    loss_history = []

    for epoch in range(1, n_epochs + 1):
        results = train_epoch(model, windows, optimizer)
        avg_loss = average_loss(results)
        loss_history.append(avg_loss)

        if verbose:
            frac = avg_loss.to_fraction()
            approx = float(avg_loss)
            print(f"Epoch {epoch:3d}: loss = {frac}  (~{approx:.6f})")

            # verify softmax sum-to-one every epoch
            for i, (loss_i, probs_i) in enumerate(results):
                prob_sum = VDR(0)
                for j in range(len(probs_i)):
                    prob_sum = prob_sum + probs_i[j]
                assert prob_sum == VDR(1), (
                    f"Epoch {epoch} window {i}: softmax sum = "
                    f"{prob_sum.to_fraction()}, expected exactly 1"
                )

    if verbose:
        print(f"\nTraining complete. {n_epochs} epochs, "
              f"{len(windows)} windows/epoch.")
        print(f"Final loss: {loss_history[-1].to_fraction()}")

    return model, loss_history


# -- convenience ----------------------------------------------------------

def quick_train(n_epochs=5):
    """
    Quick training run for testing.

    O: (model, losses)
    """
    return train(n_epochs=n_epochs, verbose=True)


def train_silent(n_epochs=None, model=None):
    """
    Train without printing.

    O: (model, losses)
    """
    return train(n_epochs=n_epochs, model=model, verbose=False)
