"""
Entry point for toy transformer LLM.

Usage:
    python run.py train
    python run.py generate
    python run.py verify
    python run.py inspect
    python run.py all
"""

import sys

from config import CORPUS, SEQ_LEN, N_EPOCHS, SEED
from data import build_dataset


def run_train(verbose=True):
    """Train model and return it with loss history."""
    from train import train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    model, losses = train(verbose=verbose)
    print()
    return model, losses


def run_generate(model):
    """Run generation demos."""
    from generate import show_generation
    from data import build_vocab, invert_vocab

    vocab = build_vocab(CORPUS)
    inv_vocab = invert_vocab(vocab)

    print("=" * 60)
    print("GENERATION")
    print("=" * 60)

    print("\n--- Greedy ---")
    show_generation(model, "the cat", 4, vocab, inv_vocab, strategy="greedy")

    print("\n--- Sampled ---")
    show_generation(model, "the cat", 4, vocab, inv_vocab, strategy="sample", seed=42)

    print("\n--- Top-k (k=2) ---")
    show_generation(model, "the cat", 4, vocab, inv_vocab, strategy="top_k", top_k=2, seed=42)

    print()


def run_verify(model):
    """Run verification suite."""
    from verify import verify_all
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    all_passed, results = verify_all(model, verbose=True)
    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        failed = [name for name, passed, _ in results if not passed]
        print("FAILED: %s" % ", ".join(failed))
    print()
    return all_passed


def run_inspect(model, losses=None):
    """Run inspection suite."""
    from inspect import full_inspection
    print("=" * 60)
    print("INSPECTION")
    print("=" * 60)
    full_inspection(model, losses)
    print()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "train":
        model, losses = run_train()

    elif mode == "generate":
        model, losses = run_train(verbose=False)
        run_generate(model)

    elif mode == "verify":
        model, _ = run_train(verbose=False)
        run_verify(model)

    elif mode == "inspect":
        model, losses = run_train(verbose=False)
        run_inspect(model, losses)

    elif mode == "all":
        model, losses = run_train()
        run_generate(model)
        passed = run_verify(model)
        run_inspect(model, losses)
        print("=" * 60)
        if passed:
            print("ALL COMPLETE — ALL TESTS PASSED")
        else:
            print("ALL COMPLETE — SOME TESTS FAILED")

    else:
        print("Usage: python run.py [train|generate|verify|inspect|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
