"""GINC in-context learning experiment with CSCG.

Replicates Table 1 from Swaminathan et al. NeurIPS 2023:
  "Schema-learning and rebinding as mechanisms of in-context
   learning and emergence"

Each row = one (k, n) configuration where:
  k = sentence length (tokens per context example)
  n = number of context examples in the prompt

Targets (Table 1, 50 clones):
  k=8, n=8  → ~97% accuracy
  k=3, n=8  → ~53% accuracy

Usage
-----
    .venv/bin/python experiments/ginc_experiment.py
    .venv/bin/python experiments/ginc_experiment.py --n_clones 10 --n_iter 50
    .venv/bin/python experiments/ginc_experiment.py --quick  # fast smoke test
"""

import argparse
import sys
import os
import time
import math
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from smcscg import CSCG
from experiments.ginc_data import GINCDataset


def evaluate_prompts(model, prompts, true_labels, desc=""):
    """Predict next token for each prompt and return accuracy."""
    correct = 0
    n = len(prompts)
    report_every = max(1, n // 10)
    t0 = time.time()
    for i, (prompt, label) in enumerate(zip(prompts, true_labels)):
        pred, _ = model.predict_next_obs(prompt)
        if pred == label:
            correct += 1
        if (i + 1) % report_every == 0:
            running_acc = correct / (i + 1)
            print(f"      {i+1:4d}/{n}  acc={running_acc:.3f}  "
                  f"({time.time()-t0:.0f}s elapsed)", end="\r", flush=True)
    print()  # newline after \r progress
    acc = correct / n
    ci = 1.96 * math.sqrt(acc * (1 - acc) / max(n, 1))
    if desc:
        print(f"    {desc}: {acc:.3f} ± {ci:.3f} ({correct}/{n})")
    return acc, ci


def run_experiment(args):
    print("=" * 65)
    print("GINC In-Context Learning Experiment")
    print("=" * 65)
    print(f"  n_clones={args.n_clones}, n_iter={args.n_iter}, "
          f"pseudocount={args.pseudocount}")
    print(f"  n_symbols={args.n_symbols}, n_values={args.n_values}, "
          f"n_slots={args.n_slots}, n_hmms={args.n_hmms}")

    # ------------------------------------------------------------------
    # 1. Generate / load GINC data
    # ------------------------------------------------------------------
    print("\n[1] Loading GINC dataset...")
    t0 = time.time()
    ds = GINCDataset(
        data_dir=args.data_dir,
        n_symbols=args.n_symbols,
        n_values=args.n_values,
        n_slots=args.n_slots,
        n_hmms=args.n_hmms,
    )
    print(f"    Done in {time.time()-t0:.1f}s")

    # Training sequences (100 docs × n_id_hmms concepts)
    print("\n[2] Generating training sequences...")
    t0 = time.time()
    train_seqs, _ = ds.get_train_sequences(
        n_docs=args.n_train_docs,
        sample_length=args.train_seq_len,
        seed=42,
    )
    total_tokens = sum(len(s) for s in train_seqs)
    print(f"    {len(train_seqs)} docs × {args.train_seq_len} tokens "
          f"= {total_tokens:,} total tokens  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Train CSCG
    # ------------------------------------------------------------------
    print(f"\n[3] Training CSCG (n_clones={args.n_clones}, "
          f"n_iter={args.n_iter}) ...")
    key = jax.random.PRNGKey(args.seed)
    model = CSCG(
        n_obs=args.n_symbols,
        n_clones=args.n_clones,
        pseudocount=args.pseudocount,
        key=key,
    )
    print(f"    n_states = {model.n_states} "
          f"({args.n_symbols} obs × {args.n_clones} clones)")

    t0 = time.time()
    model, lls = model.fit(
        train_seqs,
        n_iter=args.n_iter,
        tol=1e-4,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print(f"    Converged in {len(lls)} iterations ({elapsed:.1f}s)")
    print(f"    Final LL: {lls[-1]:.2f}")

    # Bits-per-symbol on a held-out doc
    test_doc = ds.get_train_sequences(n_docs=1, sample_length=1024, seed=99)[0][0]
    bps = model.bps(test_doc)
    print(f"    Test BPS (1K token doc): {bps:.4f}")

    # ------------------------------------------------------------------
    # 3. Evaluate in-context learning accuracy
    # ------------------------------------------------------------------
    print("\n[4] Evaluating in-context next-token prediction...")

    ks = args.sentence_lens
    ns = args.n_context_list

    # Header
    print(f"\n{'k':>4} {'n':>6}  {'Accuracy':>10}  {'95% CI':>10}")
    print("-" * 40)

    results = {}
    for k in ks:
        for n in ns:
            if (k + 1) * (n + 1) > 1024:
                print(f"{k:>4} {n:>6}  {'(too long)':>10}")
                continue

            print(f"\n  k={k}, n={n} — generating {args.n_prompts} prompts...")
            t0 = time.time()
            prompts, true_labels = ds.get_test_prompts(
                sentence_len=k,
                n_context_sentences=n,
                n_prompts=args.n_prompts,
                seed=1114,
            )
            print(f"    prompt len={len(prompts[0])} tokens  "
                  f"({time.time()-t0:.1f}s to generate)")
            acc, ci = evaluate_prompts(model, prompts, true_labels,
                                       desc=f"k={k} n={n}")
            results[(k, n)] = (acc, ci)
            print(f"  → k={k:2d} n={n:3d}  acc={acc:.3f} ±{ci:.3f}")

    print("\n[5] Summary table")
    print(f"\n{'Model':<20} {'k':<4} {'n':<6} {'Acc':>8} {'CI':>8}")
    print("-" * 50)
    model_name = f"CSCG-{args.n_clones}cl"
    for (k, n), (acc, ci) in sorted(results.items()):
        print(f"{model_name:<20} {k:<4} {n:<6} {acc:>8.3f} ±{ci:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GINC CSCG experiment")
    # Device
    parser.add_argument("--platform", type=str, default="cpu",
                        choices=["cpu", "gpu", "tpu"],
                        help="JAX platform (default: cpu)")
    # CSCG params
    parser.add_argument("--n_clones", type=int, default=50)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--pseudocount", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    # Data params (default = paper values from data dir name)
    parser.add_argument("--n_symbols", type=int, default=50)
    parser.add_argument("--n_values", type=int, default=10)
    parser.add_argument("--n_slots", type=int, default=10)
    parser.add_argument("--n_hmms", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data/ginc")
    # Training data
    parser.add_argument("--n_train_docs", type=int, default=100,
                        help="documents per concept for training")
    parser.add_argument("--train_seq_len", type=int, default=10240)
    # Evaluation
    parser.add_argument("--sentence_lens", type=int, nargs="+",
                        default=[3, 5, 8, 10])
    parser.add_argument("--n_context_list", type=int, nargs="+",
                        default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--n_prompts", type=int, default=2500)
    parser.add_argument("--verbose", action="store_true")
    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: fewer docs, 5 EM iters, 50 prompts")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", args.platform)

    if args.quick:
        args.n_iter = 5
        args.n_train_docs = 2
        args.train_seq_len = 1000
        args.n_prompts = 50
        args.sentence_lens = [8]
        args.n_context_list = [0, 4, 8]
        args.n_clones = 10
        args.verbose = True

    run_experiment(args)


if __name__ == "__main__":
    main()
