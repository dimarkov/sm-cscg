#!/usr/bin/env python3
"""Example 1: Abstract symbol patterns with aliased observations.

Ground-truth graph has 7 unique hidden states but only 6 observations,
creating an aliasing problem:

  Hidden:  H0(A) -> H1(D) -> H2(E)     Path 1
  Hidden:  H3(B) -> H4(D) -> H5(F)     Path 2
  Hidden:  H6(C) -> ...                 connects paths

Observations: A=0, B=1, C=2, D=3, E=4, F=5
Observation D is aliased: it appears in two different contexts.

Two modes:
  1. No duration: each state emits once per visit.
  2. With duration: each state emits for a Poisson-sampled duration,
     and the duration depends on the hidden context.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import jax
from smcscg import CSCG, SMCSCG, metrics


# Observation mapping
OBS_NAMES = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}
N_OBS = 6


def generate_sequences(n_sequences=80, add_duration=False, rng=None):
    """Generate sequences from a ground-truth graph with aliased observations.

    Returns
    -------
    sequences : list of ndarray
    gt_segments : list of list of (obs, start, duration) if add_duration else None
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Duration parameters per hidden state (only used if add_duration=True)
    # H0(A):3, H1(D-ctx1):5, H2(E):4, H3(B):2, H4(D-ctx2):2, H5(F):3, H6(C):3
    dur_params = {
        "H0": 3.0,  # A
        "H1": 5.0,  # D in context of A
        "H2": 4.0,  # E
        "H3": 2.0,  # B
        "H4": 2.0,  # D in context of B
        "H5": 3.0,  # F
        "H6": 3.0,  # C (connector)
    }
    state_obs = {
        "H0": 0, "H1": 3, "H2": 4,  # Path 1: A, D, E
        "H3": 1, "H4": 3, "H5": 5,  # Path 2: B, D, F
        "H6": 2,                      # Connector: C
    }

    # Paths through the graph
    paths = [
        ["H6", "H0", "H1", "H2"],  # C -> A -> D -> E
        ["H6", "H3", "H4", "H5"],  # C -> B -> D -> F
    ]

    sequences = []
    gt_segments_list = [] if add_duration else None

    for _ in range(n_sequences):
        obs_list = []
        gt_segs = [] if add_duration else None
        # Generate 3-5 path traversals per sequence
        n_traversals = rng.integers(3, 6)
        for _ in range(n_traversals):
            path = paths[rng.integers(0, 2)]
            for hidden_state in path:
                obs = state_obs[hidden_state]
                if add_duration:
                    dur = max(1, rng.poisson(dur_params[hidden_state] - 1) + 1)
                    start = len(obs_list)
                    gt_segs.append((obs, start, dur))
                    obs_list.extend([obs] * dur)
                else:
                    obs_list.append(obs)

        sequences.append(np.array(obs_list, dtype=np.intp))
        if add_duration:
            gt_segments_list.append(gt_segs)

    return sequences, gt_segments_list


def analyze_clone_specialization(model, sequences, obs_to_check=3):
    """Check if different clones of a given observation activate in different contexts."""
    print(f"\n  Clone specialization for observation {OBS_NAMES[obs_to_check]} "
          f"(obs index {obs_to_check}):")

    clone_contexts = {}  # clone_id -> list of (prev_obs, next_obs)

    for seq in sequences[:20]:
        states = model.decode(seq)
        if isinstance(states, tuple):
            states = states[0]
        for t in range(len(seq)):
            if seq[t] == obs_to_check:
                clone_id = states[t]
                prev_obs = seq[t - 1] if t > 0 else -1
                next_obs = seq[t + 1] if t < len(seq) - 1 else -1
                if clone_id not in clone_contexts:
                    clone_contexts[clone_id] = []
                clone_contexts[clone_id].append((prev_obs, next_obs))

    for clone_id, contexts in sorted(clone_contexts.items()):
        prev_counts = {}
        for prev_obs, _ in contexts:
            name = OBS_NAMES.get(prev_obs, "START")
            prev_counts[name] = prev_counts.get(name, 0) + 1
        print(f"    Clone {clone_id}: preceded by {prev_counts} ({len(contexts)} occurrences)")


def run_no_duration():
    """Part 1: sequences without duration, testing aliasing resolution."""
    print("=" * 60)
    print("PART 1: Symbol patterns WITHOUT duration")
    print("=" * 60)

    rng = np.random.default_rng(42)
    sequences, _ = generate_sequences(n_sequences=80, add_duration=False, rng=rng)

    train_seqs = sequences[:60]
    test_seqs = sequences[60:]

    print(f"\n  Generated {len(train_seqs)} train, {len(test_seqs)} test sequences")
    print(f"  Average length: {np.mean([len(s) for s in train_seqs]):.1f}")
    print(f"  Sample: {[OBS_NAMES[o] for o in train_seqs[0][:15]]}")

    # Train CSCG
    print("\n  Training CSCG (n_clones=5)...")
    cscg, lls = CSCG(n_obs=N_OBS, n_clones=5, key=jax.random.PRNGKey(123)).fit(
        train_seqs, n_iter=40, verbose=False
    )
    print(f"    Final train LL: {lls[-1]:.2f} ({len(lls)} iterations)")

    train_bps = metrics.bits_per_symbol(cscg, train_seqs)
    test_bps = metrics.bits_per_symbol(cscg, test_seqs)
    print(f"    Train BPS: {train_bps:.4f}")
    print(f"    Test  BPS: {test_bps:.4f}")

    analyze_clone_specialization(cscg, test_seqs, obs_to_check=3)

    return {"model": "CSCG", "dataset": "symbols-no-dur", "n_clones": 3,
            "train_bps": train_bps, "test_bps": test_bps}


def run_with_duration():
    """Part 2: sequences with duration, comparing CSCG vs SM-CSCG."""
    print("\n" + "=" * 60)
    print("PART 2: Symbol patterns WITH duration")
    print("=" * 60)

    rng = np.random.default_rng(42)
    sequences, gt_segments = generate_sequences(
        n_sequences=60, add_duration=True, rng=rng
    )

    train_seqs = sequences[:45]
    test_seqs = sequences[45:]
    gt_seg_test = gt_segments[45:]

    print(f"\n  Generated {len(train_seqs)} train, {len(test_seqs)} test sequences")
    print(f"  Average length: {np.mean([len(s) for s in train_seqs]):.1f}")
    sample = train_seqs[0][:30]
    print(f"  Sample: {''.join(OBS_NAMES[o] for o in sample)}")

    results = []

    # --- CSCG ---
    print("\n  Training CSCG (n_clones=5) on duration data...")
    cscg, lls = CSCG(n_obs=N_OBS, n_clones=5, key=jax.random.PRNGKey(123)).fit(
        train_seqs, n_iter=40, verbose=False
    )
    print(f"    Final train LL: {lls[-1]:.2f}")

    train_bps = metrics.bits_per_symbol(cscg, train_seqs)
    test_bps = metrics.bits_per_symbol(cscg, test_seqs)
    print(f"    Train BPS: {train_bps:.4f}")
    print(f"    Test  BPS: {test_bps:.4f}")

    results.append({"model": "CSCG", "dataset": "symbols-dur", "n_clones": 3,
                     "train_bps": train_bps, "test_bps": test_bps})

    # --- SM-CSCG ---
    print("\n  Training SM-CSCG (n_clones=5, n_phases=8, coxian) on duration data...")
    smcscg, lls = SMCSCG(n_obs=N_OBS, n_clones=5, n_phases=8,
                          phase_type="coxian", key=jax.random.PRNGKey(123)).fit(
        train_seqs, n_iter=30, verbose=False
    )
    print(f"    Final train LL: {lls[-1]:.2f}")

    train_bps_sm = metrics.bits_per_symbol(smcscg, train_seqs)
    test_bps_sm = metrics.bits_per_symbol(smcscg, test_seqs)
    print(f"    Train BPS: {train_bps_sm:.4f}")
    print(f"    Test  BPS: {test_bps_sm:.4f}")

    # Segment recovery
    print("\n  Evaluating segment recovery on test sequences...")
    avg_f1 = 0.0
    for i, seq in enumerate(test_seqs):
        _, pred_segs = smcscg.decode(seq)
        # Convert gt_segments to (state, start, dur) format for metrics
        gt_segs = [(s[0], s[1], s[2]) for s in gt_seg_test[i]]
        _, _, f1 = metrics.segment_f1(pred_segs, gt_segs, tolerance=2)
        avg_f1 += f1
    avg_f1 /= len(test_seqs)
    print(f"    Average segment boundary F1: {avg_f1:.3f}")

    results.append({"model": "SM-CSCG", "dataset": "symbols-dur", "n_clones": 3,
                     "train_bps": train_bps_sm, "test_bps": test_bps_sm,
                     "seg_f1": avg_f1})

    analyze_clone_specialization(smcscg, test_seqs, obs_to_check=3)

    # Show learned durations for D-clones
    print("\n  Learned duration distributions for D-clones (phase-type implied):")
    # D = observation 3; macro-state indices for D clones
    d_macro_clones = list(range(3 * smcscg.n_clones, 4 * smcscg.n_clones))
    for j in d_macro_clones:
        pmf = smcscg.duration_pmf(j, max_d=20)
        ds = np.arange(1, len(pmf) + 1)
        mean_d = np.dot(ds, pmf) / max(pmf.sum(), 1e-12)
        print(f"    Macro-state {j}: mean duration = {mean_d:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="cpu",
                        choices=["cpu", "gpu", "tpu"],
                        help="JAX platform (default: cpu)")
    args = parser.parse_args()
    jax.config.update("jax_platform_name", args.platform)

    r1 = run_no_duration()
    r2_list = run_with_duration()

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    metrics.print_comparison_table([r1] + r2_list)


if __name__ == "__main__":
    main()
