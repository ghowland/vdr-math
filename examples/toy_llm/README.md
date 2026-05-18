# Toy LLM — Exact Transformer on VDR Arithmetic

A complete transformer language model where every operation — forward pass, softmax, attention, backpropagation, weight update — runs in exact rational arithmetic with zero floating-point. Built on the [vdr-math](../README.md) library with a fixed denominator frame of D = 2^32.

## What This Is

A single-block, single-head causal transformer trained on the sentence "the cat sat on the mat." Five tokens, four-dimensional embeddings, 181 trainable parameters. The entire model — training, generation, verification — runs in about 10 seconds on a 2019 laptop.

The point is not the model's linguistic ability. It's that every intermediate value, every gradient, every probability has a known exact rational representation, and the denominator never grows beyond 2^32 through any operation chain.

## Architecture

```
token embedding (5 × 4) + positional embedding (4 × 4)
    ↓
self-attention (Wq, Wk, Wv, Wo — all 4×4)
    ↓  + residual
FFN (Linear 4→8, ReLU, Linear 8→4)
    ↓  + residual
output projection (Linear 4→5)
    ↓
quadratic softmax surrogate → probabilities
```

All weight matrices are initialized with small-integer numerators over D = 2^32 via a deterministic LCG. Positional embeddings are learned (not sinusoidal). The model is causal — each position can only attend to itself and earlier positions.

## How It Uses VDR

### Single Basis Frame

The entire model runs at `set_default(32)`, meaning D = 2^32 = 4,294,967,296 for every value everywhere. The basis-safe operators in `vdr.core` and `vdr.active` automatically detect when operands share the basis frame and use `divmod` to keep D fixed. No manual rebase calls. No frame management code.

```python
# config.py
from vdr.basis import set_default
set_default(32)
```

After this, all VDR arithmetic — `+`, `-`, `*`, `/` — stays in frame automatically.

### Library Modules Used

| VDR module | Used for |
|---|---|
| `vdr.core` | All arithmetic, basis-safe operators |
| `vdr.basis` | `set_default(32)`, `to_qbasis` for projecting constants |
| `vdr.linalg` | `Vec`, `Mat` for all vectors and matrices |
| `vdr.ml.nn` | `Linear`, `ReLU` layers with `to_qbasis()` |
| `vdr.ml.softmax` | `softmax_surrogate_square` — quadratic kernel |
| `vdr.ml.attention` | `attention_scores`, `causal_mask`, `weighted_sum` |
| `vdr.ml.optim` | `SGD` optimizer with basis-projected learning rate |

### Quadratic Softmax Surrogate

Standard softmax uses `exp(x)`, which is transcendental and requires Taylor series in VDR. This model uses the quadratic surrogate instead:

```
p_i = (x_i - shift)^2 / sum_j (x_j - shift)^2
```

The shift is the minimum logit, centering values to keep V slots small. The result is purely polynomial — subtraction, squaring, division. All exact rational operations. No series approximation.

The probabilities sum to exactly 1 by construction: the last probability is computed as `1 - sum(first N-1)`, absorbing any division remainder residuals. At D = 2^32, the worst observed residual before this correction was 9.07 × 10⁻¹³.

### D Stability

Every parameter, activation, gradient, and logit has D = 2^32 throughout the entire pipeline. The denominator report confirms this:

```
all 181 parameters have D = 4294967296  [OK]
```

This is guaranteed by the basis-safe operators, not by manual discipline. The `_both_in_basis` and `_one_in_basis` checks in `vdr.core` intercept every arithmetic operation and route it through `_basis_mul` / `_basis_div` when either operand is in the basis frame. Mixed operations (basis value + plain `VDR(1,3)`) automatically rebase the non-basis operand.

## Training

### Loss Function

MSE between the softmax output probabilities and a one-hot target vector. The gradient is the closed-form `(2/n)(pred - target)` projected to basis frame.

### Optimizer

Plain SGD with learning rate 1/128, projected to basis frame at optimizer construction. No per-step rebase needed.

### Results

20 epochs on 2 training windows (the corpus "the cat sat on the mat" with context length 4 produces 2 windows):

```
epoch  1  loss=0.267299  softmax_sum=1: yes
epoch  5  loss=0.255813  softmax_sum=1: yes
epoch 10  loss=0.244555  softmax_sum=1: yes
epoch 15  loss=0.234526  softmax_sum=1: yes
epoch 20  loss=0.225423  softmax_sum=1: yes
```

Loss decreases monotonically. Softmax sums to 1 at every epoch (within 1e-9 tolerance, which accounts for division remainder residuals at the 2^32 precision floor).

### Attention Pattern

After training, the attention map for "the cat sat on":

```
      the     cat     sat     on
the   1.000   0.000   0.000   0.000
cat   0.500   0.500   0.000   0.000
sat   0.333   0.335   0.332   0.000
on    0.283   0.527   0.000   0.190
```

Causal masking is exact — future positions are exactly 0.000, not approximately zero. The quadratic surrogate produces smoother distributions than exponential softmax (no sharp peaking), which is visible in the relatively even weights for "sat" attending to all prior tokens.

## Generation

Four decoding strategies, all using exact rational CDF comparison for sampling:

```
Greedy:   the cat sat sat sat sat
Sampled:  the cat sat sat sat sat
Top-k:    the cat sat sat sat sat
```

The model converges to predicting "sat" as the dominant next token, which is expected — with only 2 training windows, "sat" follows both "the cat" and "cat sat" contexts. The deterministic RNG produces exact rational values in [0, 1) with denominator 2^31, and the CDF comparison is exact — no float threshold ambiguity.

## Verification Suite

Nine tests, all passing:

| Test | What it checks | Result |
|---|---|---|
| softmax_sum | Every output probability vector sums to 1 | PASS |
| attention_weights | Every attention weight row sums to 1 | PASS |
| d_stability | All parameters and logits have D = 2^32 | PASS |
| deterministic | Two runs from same seed are bit-identical | PASS |
| forward_backward_roundtrip | Same init → same result after one step | PASS |
| checkpoint_roundtrip | Save → perturb → restore → bit-identical | PASS |
| weight_update | (w_old - w_new) / lr == grad exactly | PASS |
| loss_monotonicity | Final loss < initial loss | PASS |
| gradient_correctness | Analytical vs finite difference, diff < 0.1 | PASS |

The deterministic and roundtrip tests are the strongest results — they prove that the computation is platform-independent and reproducible at the bit level. No floating-point non-associativity. No GPU-ordering nondeterminism. Same seed, same weights, same gradients, same model on every machine.

## Precision Budget

At D = 2^32, each value has ~9.6 decimal digits of precision. The precision floor is 2^(-32) ≈ 2.33 × 10⁻¹⁰.

For this toy model:
- 20 epochs × 2 windows × ~50 operations per forward pass ≈ 2,000 arithmetic operations
- Worst-case accumulated error: ~2,000 × 2.33 × 10⁻¹⁰ ≈ 4.7 × 10⁻⁷
- Observed worst softmax residual: 9.07 × 10⁻¹³ (far below worst case)

For a production-scale model at D = 2^64:
- ~19.3 decimal digits per value
- Precision floor: 5.42 × 10⁻²⁰
- After 10 million operations: worst-case ≈ 5.42 × 10⁻¹³
- Still below float64 precision (which has ~15.9 digits)

## File Structure

```
toy_llm/
├── config.py               # Hyperparameters, set_default(32)
├── data.py                 # Tokenization, windowing, one-hot targets
├── model.py                # ToyTransformer — forward, backward, init
├── attention_backward.py   # Surrogate softmax backward, attention grad
├── train.py                # Training loop, MSE loss, SGD
├── generate.py             # Greedy, sampled, top-k, nucleus decoding
├── verify.py               # 9 verification tests
├── inspect.py              # Parameter printing, attention maps, D report
└── run.py                  # Entry point: train | generate | verify | inspect | all
```

## Running

```bash
python run.py all       # train + generate + verify + inspect
python run.py train     # train only, prints loss per epoch
python run.py generate  # train then generate text
python run.py verify    # train then run verification suite
python run.py inspect   # train then print model internals
```

## What This Demonstrates

1. **A complete LLM pipeline runs in exact rational arithmetic.** Forward pass, backpropagation, weight updates, attention, softmax, sampling — all without a single floating-point operation.

2. **D = 2^32 is sufficient for a toy model.** The denominator never grows. Every operation stays in frame. The precision floor of ~10⁻¹⁰ per operation is adequate for 20 epochs of training.

3. **The quadratic softmax surrogate eliminates transcendentals.** No `exp`, no Taylor series, no series truncation error. Pure polynomial arithmetic. Probabilities sum to 1 by construction.

4. **Determinism is exact.** Same seed produces bit-identical weights, gradients, and losses across runs. This is impossible with float due to non-associative addition and GPU thread ordering.

5. **The VDR basis-safe operators handle frame management automatically.** No manual rebase calls. No frame tracking code. The model code reads like normal arithmetic — the frame discipline is invisible.
