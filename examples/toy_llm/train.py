"""
Training loop for toy transformer LLM.
MSE loss on surrogate softmax output vs one-hot target.
All arithmetic exact VDR in basis frame.
"""

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.basis import to_qbasis
from vdr.ml.optim import SGD
from vdr.ml.softmax import softmax_surrogate_square

from config import VOCAB_SIZE, N_EPOCHS, LR, SEED, CORPUS, SEQ_LEN
from data import build_dataset, one_hot_target


def mse_loss(pred, target):
    """
    Mean squared error.

    I: pred Vec, target Vec
    O: VDR scalar, exact
    """
    n = len(pred)
    inv_n = to_qbasis(VDR(1, n))
    total = VDR(0)
    for i in range(n):
        diff = pred[i] - target[i]
        total = total + diff * diff
    return total * inv_n


def mse_loss_grad(pred, target):
    """
    Gradient of MSE w.r.t. pred.

    I: pred Vec, target Vec
    O: Vec gradient, exact
    """
    n = len(pred)
    two_over_n = to_qbasis(VDR(2, n))
    result = []
    for i in range(n):
        result.append(two_over_n * (pred[i] - target[i]))
    return Vec(result)


def surrogate_softmax_then_mse(logits, target_vec):
    """
    Surrogate softmax -> MSE loss pipeline.

    I: logits Vec, target one-hot Vec
    O: (probs Vec summing to 1, loss VDR, grad_logits Vec)

    Uses quadratic surrogate, not Taylor exp.
    """
    # softmax surrogate with min-shift
    shift = logits[0]
    for i in range(1, len(logits)):
        if logits[i] < shift:
            shift = logits[i]
    probs = softmax_surrogate_square(logits, shift=shift)

    loss = mse_loss(probs, target_vec)
    grad_probs = mse_loss_grad(probs, target_vec)

    # backward through surrogate softmax
    shifted = Vec([x - shift for x in logits])
    n = len(probs)
    sum_sq = VDR(0)
    for i in range(n):
        sum_sq = sum_sq + shifted[i] * shifted[i]

    if sum_sq == VDR(0):
        grad_logits = Vec([VDR(0)] * n)
    else:
        grad_logits_data = []
        dot_gp = VDR(0)
        for j in range(n):
            dot_gp = dot_gp + grad_probs[j] * probs[j]
        two = to_qbasis(VDR(2))
        for i in range(n):
            g = two * shifted[i] / sum_sq * (grad_probs[i] - dot_gp)
            grad_logits_data.append(g)
        grad_logits = Vec(grad_logits_data)

    return probs, loss, grad_logits


def train_step(model, context_ids, target_id, optimizer):
    """
    Single training step.

    I: model, context token ids, target token id, optimizer
    O: (loss VDR, probs Vec)
    S: forward -> loss -> backward -> step
    """
    target_vec = one_hot_target(target_id, VOCAB_SIZE)

    # forward with cache
    logits, cache = model.forward_with_cache(context_ids)
    last_logits = logits[-1]

    # loss on last position
    probs, loss, grad_logits = surrogate_softmax_then_mse(last_logits, target_vec)

    # backward from last position
    model.zero_grad()
    model.backward_from_last(grad_logits)

    # optimizer step
    optimizer.step()

    return loss, probs


def train_epoch(model, windows, optimizer):
    """
    Train one epoch over all windows.

    I: model, list of (context_ids, target_id), optimizer
    O: list of (loss, probs) per window
    """
    results = []
    for context_ids, target_id in windows:
        loss, probs = train_step(model, context_ids, target_id, optimizer)
        results.append((loss, probs))
    return results


def average_loss(results):
    """
    Average loss over epoch results.

    I: list of (loss, probs)
    O: VDR average loss
    """
    if not results:
        return VDR(0)
    total = VDR(0)
    for loss, _ in results:
        total = total + loss
    return total * to_qbasis(VDR(1, len(results)))


def train(n_epochs=None, model=None, verbose=True):
    """
    Full training loop.

    I: epochs (default from config), optional model, verbose flag
    O: (trained model, list of average loss per epoch)
    """
    if n_epochs is None:
        n_epochs = N_EPOCHS

    from model import ToyTransformer
    if model is None:
        model = ToyTransformer()

    windows, vocab, inv_vocab, vocab_size = build_dataset(CORPUS, SEQ_LEN)
    optimizer = SGD(model.parameters(), lr=LR)

    loss_history = []

    for epoch in range(n_epochs):
        results = train_epoch(model, windows, optimizer)
        avg = average_loss(results)
        loss_history.append(avg)

        if verbose:
            # check softmax sum
            sum_ok = True
            for _, probs in results:
                total = VDR(0)
                for i in range(len(probs)):
                    total = total + probs[i]
                if total != to_qbasis(VDR(1)):
                    sum_ok = False
                    break

            print("  epoch %2d  loss=%.6f  softmax_sum=1: %s" % (
                epoch + 1, avg.to_float(), "yes" if sum_ok else "NO"))

    return model, loss_history


def quick_train(n_epochs=5):
    """Train with fewer epochs, verbose."""
    return train(n_epochs=n_epochs, verbose=True)


def train_silent(n_epochs=None, model=None):
    """Train without printing."""
    return train(n_epochs=n_epochs, model=model, verbose=False)
