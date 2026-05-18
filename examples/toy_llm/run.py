"""
examples/toy_llm/run.py

Main entry point for the small-frame toy LLM.

    python run.py              # train + generate + verify
    python run.py train        # train only
    python run.py generate     # train + generate
    python run.py verify       # train + verify
    python run.py inspect      # train + full inspection
    python run.py all          # everything

Small fixed D-frames: weights D=128, activations D=128,
scores D=256, probabilities D=256, gradients D=32768.
Denominators never grow. Runtime: seconds, not minutes.
"""

import sys

from data import build_dataset
from model import ToyTransformer
from train import train
from generate import generate_greedy, generate_sampled, show_generation
from verify import verify_all
from inspect_model import full_inspection, print_loss_trajectory
import config as cfg


def run_train(verbose=True):
    """Train the model. Return (model, losses)."""
    print("=" * 60)
    print("VDR Toy LLM — Training (Small Frames)")
    print(f"  Vocab size:  {cfg.VOCAB_SIZE}")
    print(f"  Dim:         {cfg.DIM}")
    print(f"  Seq len:     {cfg.SEQ_LEN}")
    print(f"  FFN dim:     {cfg.FFN_DIM}")
    print(f"  LR:          {cfg.LR.to_fraction()}")
    print(f"  Epochs:      {cfg.N_EPOCHS}")
    print(f"  Seed:        {cfg.SEED}")
    print(f"  D_WEIGHT:    {cfg.D_WEIGHT}")
    print(f"  D_ACT:       {cfg.D_ACT}")
    print(f"  D_SCORE:     {cfg.D_SCORE}")
    print(f"  D_PROB:      {cfg.D_PROB}")
    print(f"  D_GRAD:      {cfg.D_GRAD}")
    print(f"  Softmax:     surrogate (rational, no transcendentals)")
    print("=" * 60)
    print()

    model, losses = train(n_epochs=cfg.N_EPOCHS, verbose=verbose)
    return model, losses


def run_generate(model):
    """Generate text with multiple strategies."""
    print()
    print("=" * 60)
    print("VDR Toy LLM — Generation")
    print("=" * 60)
    print()

    print("--- Greedy ---")
    text = generate_greedy(model, "the cat", max_tokens=4)
    print(f"  '{text}'")
    print()

    print("--- Sampled (seed=1) ---")
    text = generate_sampled(model, "the cat", max_tokens=4, seed=1)
    print(f"  '{text}'")
    print()

    print("--- Sampled (seed=42) ---")
    text = generate_sampled(model, "the cat", max_tokens=4, seed=42)
    print(f"  '{text}'")
    print()

    print("--- Detailed generation ---")
    show_generation(model, "the cat", max_tokens=4, strategy="greedy")
    print()


def run_verify(model):
    """Run verification suite."""
    print()
    passed, results = verify_all(model, verbose=True)
    return passed


def run_inspect(model, losses):
    """Run full inspection."""
    print()
    full_inspection(model, losses)


def main():
    """Main entry point."""
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    mode = args[0].lower()

    if mode == "train":
        model, losses = run_train()
        print_loss_trajectory(losses)

    elif mode == "generate":
        model, losses = run_train()
        run_generate(model)

    elif mode == "verify":
        model, losses = run_train()
        run_verify(model)

    elif mode == "inspect":
        model, losses = run_train()
        run_inspect(model, losses)

    elif mode in ("all", "full"):
        model, losses = run_train()
        run_generate(model)
        passed = run_verify(model)
        run_inspect(model, losses)

        print()
        print("=" * 60)
        if passed:
            print("ALL COMPLETE. Every value exact. Frames bounded. "
                  "All tests passed.")
        else:
            print("COMPLETE WITH FAILURES. See verification output.")
        print("=" * 60)

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python run.py [train|generate|verify|inspect|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()

