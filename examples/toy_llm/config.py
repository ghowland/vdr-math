"""
Toy LLM configuration — all hyperparameters and frame constants.
Single basis frame: 2^32 everywhere via set_default(32).
"""

from vdr.core import VDR
from vdr.basis import set_default, get_default, to_qbasis

# Set global basis frame
set_default(32)

# Model shape
VOCAB_SIZE = 5
DIM = 4
SEQ_LEN = 4
FFN_DIM = 8
N_HEADS = 1

# Training
N_EPOCHS = 20
LR = to_qbasis(VDR(1, 128))
SEED = 42

# Corpus
CORPUS = "the cat sat on the mat"
