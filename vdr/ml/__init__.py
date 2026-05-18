"""
vdr.ml — Exact machine learning primitives.

    from vdr.ml.softmax import softmax
    from vdr.ml.nn import Linear, ReLU, Sequential
    from vdr.ml.attention import self_attention
    from vdr.ml.optim import SGD

Every operation exact VDR rational. Softmax sums to exactly 1.
Gradients exact via chain rule. No float drift in training loops.
"""
