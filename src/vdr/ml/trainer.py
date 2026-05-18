"""
vdr.ml.trainer — Exact training loops for VDR neural networks.

    from vdr.ml.trainer import train_step, train_epoch, evaluate_classification

    loss = train_step(model, x, y, optimizer)
    losses = train_epoch(model, dataset, optimizer)
    accuracy = evaluate_classification(model, dataset)

Every forward pass, gradient, and parameter update exact VDR rational.
Loss values are exact fractions, not approximate floats.
"""

from __future__ import annotations
from typing import List, Tuple, Sequence, Callable

from vdr.core import VDR
from vdr.linalg import Vec
from vdr.ml.losses import mse, mse_grad

__all__ = [
    "train_step",
    "train_epoch",
    "evaluate_epoch",
    "predict_class",
    "evaluate_classification",
]


def train_step(model, x, y, optimizer):
    """
    One training step: forward, loss, backward, update.

    I: model (Module with forward/backward), input x (Vec),
       target y (Vec), optimizer (SGD or Momentum)
    O: loss value as VDR, exact

    S: updates model parameters via optimizer

        loss = train_step(model, x, y, opt)
    """
    # forward
    pred = model.forward(x)

    # loss
    loss = mse(pred, y)

    # gradient of loss w.r.t. output
    grad = mse_grad(pred, y)

    # backward
    model.backward(grad)

    # update
    optimizer.step()

    # zero grads for next step
    optimizer.zero_grad()

    return loss


def train_epoch(model, dataset, optimizer):
    """
    Train on entire dataset for one epoch.

    I: model, dataset as list of (x, y) tuples, optimizer
    O: list of per-sample loss values (VDR)

        losses = train_epoch(model, [(x1,y1), (x2,y2), ...], opt)
    """
    losses = []
    for x, y in dataset:
        loss = train_step(model, x, y, optimizer)
        losses.append(loss)
    return losses


def evaluate_epoch(model, dataset):
    """
    Evaluate model on dataset without updating parameters.

    I: model, dataset as list of (x, y) tuples
    O: list of per-sample loss values (VDR)

        eval_losses = evaluate_epoch(model, test_set)
    """
    losses = []
    for x, y in dataset:
        pred = model.forward(x)
        loss = mse(pred, y)
        losses.append(loss)
    return losses


def predict_class(model, x):
    """
    Predict class label by argmax of model output.

    I: model, input x (Vec)
    O: integer class index

        label = predict_class(model, x)
    """
    output = model.forward(x)
    best_idx = 0
    best_val = output[0]
    for i in range(1, len(output)):
        if output[i] > best_val:
            best_val = output[i]
            best_idx = i
    return best_idx


def evaluate_classification(model, dataset):
    """
    Evaluate classification accuracy on dataset.

    I: model, dataset as list of (x, y) tuples where y encodes target class
       (either as one-hot Vec or as int)
    O: accuracy as VDR (correct / total), exact rational

        accuracy = evaluate_classification(model, test_set)
        # e.g. VDR(7, 10) = 70% accuracy
    """
    correct = 0
    total = 0

    for x, y in dataset:
        pred_class = predict_class(model, x)

        # determine target class
        if isinstance(y, Vec):
            # one-hot: find argmax
            target_class = 0
            best = y[0]
            for i in range(1, len(y)):
                if y[i] > best:
                    best = y[i]
                    target_class = i
        elif isinstance(y, int):
            target_class = y
        else:
            target_class = int(y.v)

        if pred_class == target_class:
            correct += 1
        total += 1

    if total == 0:
        return VDR(0)
    return VDR(correct, total)
