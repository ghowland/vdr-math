"""
examples/toy_llm/config.py

Toy LLM configuration — small fixed D-frames matching frontier quantized models.
Every numeric value is a VDR or int. No float.

D-frames are power-of-two denominators:
  D_WEIGHT = 128   (i8-scale weights)
  D_ACT    = 128   (i8-scale activations)
  D_SCORE  = 256   (attention scores)
  D_PROB   = 256   (softmax probabilities)
  D_ACC    = 32768 (i16-scale accumulators)
  D_GRAD   = 32768 (i16-scale gradients)
"""

from vdr.core import VDR

# D-frames — power-of-two denominators, divmod is bit-shift
D_WEIGHT = 128
D_ACT = 128
D_SCORE = 256
D_PROB = 256
D_ACC = 32768
D_GRAD = 32768
D_OPT = 32768

# Model shape
VOCAB_SIZE = 5
DIM = 4
SEQ_LEN = 4
FFN_DIM = 8
N_HEADS = 1

# Training
N_EPOCHS = 20
LR = VDR(1, 128)       # learning rate in weight frame
SEED = 42

# Surrogate softmax shift constant
SURR_C = VDR(4, 1)     # additive shift to ensure positivity

CORPUS = "the cat sat on the mat"
