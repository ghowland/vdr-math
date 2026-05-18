"""
examples/toy_llm/config.py

Toy LLM configuration — small fixed D-frames.
"""

from vdr.core import VDR

# D-frames as bit counts — D = 2^bits
BITS_WEIGHT = 7       # D=128, i8-scale weights
BITS_ACT = 7          # D=128, i8-scale activations
BITS_SCORE = 8        # D=256, attention scores
BITS_PROB = 8         # D=256, softmax probabilities
BITS_GRAD = 15        # D=32768, i16-scale gradients

# Derived D values
D_WEIGHT = 2 ** BITS_WEIGHT   # 128
D_ACT = 2 ** BITS_ACT         # 128
D_SCORE = 2 ** BITS_SCORE     # 256
D_PROB = 2 ** BITS_PROB       # 256
D_GRAD = 2 ** BITS_GRAD       # 32768

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

# Surrogate softmax shift
SURR_C = VDR(4, 1)

CORPUS = "the cat sat on the mat"
