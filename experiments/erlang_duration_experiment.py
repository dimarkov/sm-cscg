#!/usr/bin/env python3
"""Erlang-k duration experiment: CSCG vs SM-CSCG for phase-type durations.

Generates binary sequences (0s and 1s) where the inter-event gap (count of
0s between consecutive 1s) follows a negative binomial NB(k, p) distribution
— the discrete analog of an Erlang-k distribution.

Compares CSCG and SM-CSCG (with n_phases=k), each trained with both
Baum-Welch and Viterbi EM, measuring:
  1. BPS (bits per symbol) — compression quality
  2. Next-token prediction accuracy — event recall via forward algorithm
  3. KL divergence — sampled vs ground-truth gap distribution (CSCG only)

Expected: SM-CSCG with n_phases=k should match CSCG performance with fewer
clones, since the phase-type structure captures duration directly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, softmax

import equinox as eqx

from smcscg import CSCG, SMCSCG


# ---------------------------------------------------------------------------
# Structured initialization for binary duration data
# ---------------------------------------------------------------------------

def make_chain_cscg(n_clones, key, continue_prob=0.7):
    """Create a CSCG(n_obs=2) with chain-structured transition matrix.

    Clones of obs 0 are arranged in a chain: clone_i -> clone_{i+1}.
    The last clone has high exit probability to obs 1.
    Obs 1 clones transition back to clone 0 of obs 0 (reset).

    Parameters
    ----------
    n_clones     : int
    key          : jax.random.PRNGKey
    continue_prob: float — probability of advancing to next clone vs exiting
    """
    C = n_clones
    N = 2 * C  # n_obs=2

    # Build transition matrix in probability space
    T = np.full((N, N), 1e-6)  # near-zero background

    # Obs 0 clones (indices 0..C-1): chain structure
    for i in range(C):
        if i < C - 1:
            # Continue to next clone with high prob, exit to obs 1 with low prob
            T[i, i + 1] = continue_prob               # advance in chain
            # Exit: spread across obs 1 clones
            for j in range(C, N):
                T[i, j] = (1.0 - continue_prob) / C
        else:
            # Last clone: must exit to obs 1
            for j in range(C, N):
                T[i, j] = 1.0 / C

    # Obs 1 clones (indices C..2C-1): reset to clone 0 of obs 0
    for i in range(C, N):
        T[i, 0] = 0.9                               # back to start of chain
        # Small prob to other obs 0 clones and obs 1 clones
        for j in range(1, N):
            if j != i:
                T[i, j] = 0.1 / (N - 2)

    # Normalize rows and convert to log-space
    T = T / T.sum(axis=1, keepdims=True)
    log_T = jnp.array(np.log(T))

    # Uniform initial distribution
    log_pi = jnp.full(N, -jnp.log(N))

    model = CSCG(n_obs=2, n_clones=n_clones, key=key)
    # Add noise to break exact symmetry between obs 1 clones
    noise = 0.05 * jax.random.uniform(key, shape=(N, N))
    log_T_noisy = jnp.log(jnp.exp(log_T) + noise)
    log_T_noisy = log_T_noisy - logsumexp(log_T_noisy, axis=1, keepdims=True)

    return eqx.tree_at(lambda m: m.log_T, model, log_T_noisy)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_deterministic_binary_sequence(gap, length):
    """Generate a periodic binary sequence: `gap` zeros then a 1, repeating.

    Parameters
    ----------
    gap    : int — number of 0s between consecutive 1s
    length : int — total sequence length

    Returns
    -------
    seq : ndarray (length,) int32
    """
    period = gap + 1
    seq = np.zeros(length, dtype=np.int32)
    seq[gap::period] = 1
    return seq


def generate_erlang_binary_sequence(k, p, length, rng):
    """Generate a binary sequence with NB(k, p)-distributed inter-event gaps.

    Parameters
    ----------
    k      : int   — Erlang shape (number of geometric stages)
    p      : float — geometric success probability per stage
    length : int   — total sequence length
    rng    : np.random.Generator

    Returns
    -------
    seq : ndarray (length,) int32
    """
    seq = []
    while len(seq) < length:
        gap = rng.negative_binomial(k, p)  # number of 0s before event
        seq.extend([0] * gap)
        seq.append(1)
    return np.array(seq[:length], dtype=np.int32)


# ---------------------------------------------------------------------------
# Gap extraction and distribution comparison
# ---------------------------------------------------------------------------

def extract_gaps(seq):
    """Extract inter-event gaps (number of 0s between consecutive 1s)."""
    ones = np.where(np.asarray(seq) == 1)[0]
    if len(ones) < 2:
        return []
    gaps = np.diff(ones) - 1  # number of 0s between consecutive 1s
    return gaps.tolist()


def gap_distribution(gaps, max_gap):
    """Convert gap list to probability distribution over [0, max_gap]."""
    if not gaps:
        return np.ones(max_gap + 1) / (max_gap + 1)
    counts = np.bincount(np.minimum(gaps, max_gap), minlength=max_gap + 1)
    total = counts.sum()
    if total == 0:
        return np.ones(max_gap + 1) / (max_gap + 1)
    return counts / total


def kl_divergence(p_dist, q_dist, eps=1e-12):
    """KL(p || q) with smoothing."""
    p = np.asarray(p_dist, dtype=np.float64) + eps
    q = np.asarray(q_dist, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def sequence_prediction_accuracy(model, obs_seq):
    """Compute event (obs=1) prediction recall via predict_next_obs.

    Works for both CSCG and SMCSCG. Only counts positions where the true
    next token is 1 (an event), avoiding inflation from the majority class.

    Returns
    -------
    recall   : float — fraction of actual 1s that were predicted as 1
    n_events : int   — number of event positions evaluated
    """
    # log_probs[t] predicts obs at t+1
    log_probs = model.predict_next_obs(obs_seq[:-1])  # (T-1, n_obs)
    true_next = jnp.asarray(obs_seq[1:], dtype=jnp.int32)

    event_mask = true_next == 1
    n_events = int(event_mask.sum())

    return softmax(log_probs[event_mask], axis=-1)[:, -1].sum() / n_events, n_events


def evaluate_model(model, test_seqs, gt_dist, max_gap, args, can_sample=True):
    """Evaluate a trained model on test data.

    Returns
    -------
    test_bps : float
    mean_acc : float — event recall
    kl       : float — KL divergence (NaN if can_sample=False)
    """
    test_bps = np.mean([model.bps(seq) for seq in test_seqs])

    recalls = []
    for seq in test_seqs:
        recall, _ = sequence_prediction_accuracy(model, seq)
        recalls.append(recall)
    mean_acc = np.mean(recalls)

    kl = float('nan')
    if can_sample:
        sampled_gaps = []
        for i in range(args.n_sample):
            sample_key = jax.random.PRNGKey(args.seed + 10000 + i)
            sampled_seq, _ = model.sample(args.sample_length, key=sample_key)
            sampled_gaps.extend(extract_gaps(sampled_seq))
        if sampled_gaps:
            sample_dist = gap_distribution(sampled_gaps, max_gap)
            kl = kl_divergence(gt_dist, sample_dist)
        else:
            kl = float('inf')

    return test_bps, mean_acc, kl


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def memoryless_entropy(k, p):
    """Entropy of a memoryless binary source with the same 0/1 frequency.

    For NB(k,p) gaps, E[D] = k*(1-p)/p, so fraction of 1s = 1/(E[D]+1).
    Returns bits per symbol.
    """
    mean_d = k * (1 - p) / p
    q = 1.0 / (mean_d + 1)  # exact fraction of 1s
    if q <= 0 or q >= 1:
        return 0.0
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def _nb_pmf(k, p, max_d=10000):
    """NB(k,p) PMF for d=0..max_d-1 via recurrence."""
    pmf = np.zeros(max_d)
    pmf[0] = p ** k
    for d in range(1, max_d):
        pmf[d] = pmf[d - 1] * (1 - p) * (d + k - 1) / d
        if pmf[d] < 1e-30:
            break
    return pmf[:d + 1] if d < max_d - 1 else pmf


def theoretical_optimal_bps(k, p):
    """Theoretical optimal BPS for a perfect model of the renewal process.

    The binary sequence is a renewal process with inter-event gaps D ~ NB(k,p).
    Each cycle is D zeros then a 1, with entropy H(D) bits.
    Entropy rate = H(NB(k,p)) / E[D+1] bits per symbol.
    """
    mean_d = k * (1 - p) / p
    pmf = _nb_pmf(k, p)
    pmf = pmf[pmf > 0]
    pmf = pmf / pmf.sum()  # renormalize after truncation
    h_nb = -np.sum(pmf * np.log2(pmf))
    return h_nb / (mean_d + 1)


def theoretical_optimal_ev_recall(k, p):
    """Theoretical optimal EvRecall for a perfect model.

    EvRecall = E_D[h(D)] = sum_d P(D=d)^2 / P(D>=d)
    where h(d) is the hazard function of NB(k, p).
    """
    pmf = _nb_pmf(k, p)
    survival = 1.0 - np.concatenate([[0.0], np.cumsum(pmf[:-1])])
    return float(np.sum(pmf ** 2 / np.maximum(survival, 1e-30)))


def run_experiment(args):
    rng = np.random.default_rng(args.seed)

    results = []
    em_methods = ["baum-welch", "viterbi"]

    # Collect all (k_label, k_phases, train_seqs, test_seqs, gt_dist, max_gap,
    #             h_memoryless, h_optimal, ev_optimal) configs
    configs = []
    gap = int(args.mean_gap)

    # --- Deterministic: fixed gap, vary n_phases ---
    q_det = 1.0 / (gap + 1)
    h_mem_det = -(q_det * np.log2(q_det) + (1 - q_det) * np.log2(1 - q_det))
    det_seq = generate_deterministic_binary_sequence(gap, args.seq_length)
    det_header = (f"Deterministic  (gap = {gap})\n"
                  f"  Memoryless baseline BPS: {h_mem_det:.4f}\n"
                  f"  Theoretical optimal BPS: 0.0000\n"
                  f"  Theoretical optimal EvRecall: 1.0000")
    for i, k in enumerate(args.erlang_shapes):
        configs.append({
            'k_label': 'det', 'k_phases': k,
            'train_seqs': [det_seq] * args.n_train,
            'test_seqs': [det_seq] * args.n_test,
            'h_memoryless': h_mem_det, 'h_optimal': 0.0, 'ev_optimal': 1.0,
            'run_cscg': i == 0,  # CSCG only on first iteration
            'header': det_header if i == 0 else None,
        })

    # --- NB(k, p) for each erlang shape ---
    for k in args.erlang_shapes:
        p = k / (args.mean_gap + k)  # E[D] = k*(1-p)/p = mean_gap
        h_memoryless = memoryless_entropy(k, p)
        h_optimal = theoretical_optimal_bps(k, p)
        ev_optimal = theoretical_optimal_ev_recall(k, p)
        train_seqs = [generate_erlang_binary_sequence(k, p,
                                                       args.seq_length, rng)
                      for _ in range(args.n_train)]
        test_seqs = [generate_erlang_binary_sequence(k, p,
                                                      args.seq_length, rng)
                     for _ in range(args.n_test)]
        configs.append({
            'k_label': k, 'k_phases': k,
            'train_seqs': train_seqs, 'test_seqs': test_seqs,
            'h_memoryless': h_memoryless, 'h_optimal': h_optimal,
            'ev_optimal': ev_optimal,
            'header': (f"Erlang shape k={k}  (mean gap = {args.mean_gap:.1f},"
                       f" p = {p:.4f})\n"
                       f"  Memoryless baseline BPS: {h_memoryless:.4f}\n"
                       f"  Theoretical optimal BPS: {h_optimal:.4f}\n"
                       f"  Theoretical optimal EvRecall: {ev_optimal:.4f}"),
        })

    for cfg in configs:
        k_label = cfg['k_label']
        k_phases = cfg['k_phases']
        train_seqs = cfg['train_seqs']
        test_seqs = cfg['test_seqs']
        h_memoryless = cfg['h_memoryless']
        h_optimal = cfg['h_optimal']
        run_cscg = cfg.get('run_cscg', True)

        if cfg.get('header'):
            print(f"\n{'='*70}")
            print(cfg['header'])
            print(f"{'='*70}")

            # Ground truth gap distribution
            all_gt_gaps = []
            for seq in train_seqs + test_seqs:
                all_gt_gaps.extend(extract_gaps(seq))
            max_gap = int(np.percentile(all_gt_gaps, 99)) + 5 if all_gt_gaps else 50
            gt_dist = gap_distribution(all_gt_gaps, max_gap)

            print(f"  Train: {len(train_seqs)} seqs x {args.seq_length} tokens")

        for n_clones in args.clone_counts:
            for em_method in em_methods:
                em_tag = "BW" if em_method == "baum-welch" else "Vit"

                # --- CSCG ---
                if run_cscg:
                    print(f"\n  CSCG    C={n_clones:>2}            {em_tag:>3}  ",
                          end="", flush=True)
                    key = jax.random.PRNGKey(
                        args.seed + hash(str(k_label)) % 10000 * 100 + n_clones)
                    model = make_chain_cscg(n_clones, key)
                    model, lls = model.fit(
                        train_seqs, n_iter=args.n_iter, tol=1e-4,
                        verbose=args.verbose, em_method=em_method
                    )
                    print(f"({len(lls):>3} it) ", end="", flush=True)

                    test_bps, mean_acc, kl = evaluate_model(
                        model, test_seqs, gt_dist, max_gap, args,
                        can_sample=True
                    )
                    results.append({
                        'k': k_label, 'model': 'CSCG', 'em': em_tag,
                        'n_clones': n_clones, 'n_phases': 1,
                        'test_bps': test_bps, 'accuracy': mean_acc,
                        'kl_divergence': kl, 'n_iters': len(lls),
                        'h_memoryless': h_memoryless,
                        'h_optimal': h_optimal,
                    })
                    print(f"BPS={test_bps:.4f}  EvRecall={mean_acc:.4f}  KL={kl:.4f}")

                # --- SM-CSCG ---
                print(f"  SMCSCG  C={n_clones:>2}  L={k_phases:>2}  {em_tag:>3}  ",
                      end="", flush=True)
                key = jax.random.PRNGKey(
                    args.seed + hash(str(k_label)) % 10000 * 100 + n_clones + 50)
                sm_model = SMCSCG(n_obs=2, n_clones=n_clones,
                                  n_phases=k_phases, key=key)
                sm_model, sm_lls = sm_model.fit(
                    train_seqs, n_iter=args.n_iter, tol=1e-4,
                    verbose=args.verbose, em_method=em_method
                )
                print(f"({len(sm_lls):>3} it) ", end="", flush=True)

                sm_bps, sm_acc, _ = evaluate_model(
                    sm_model, test_seqs, gt_dist, max_gap, args,
                    can_sample=False
                )
                results.append({
                    'k': k_label, 'model': 'SMCSCG', 'em': em_tag,
                    'n_clones': n_clones, 'n_phases': k_phases,
                    'test_bps': sm_bps, 'accuracy': sm_acc,
                    'kl_divergence': float('nan'), 'n_iters': len(sm_lls),
                    'h_memoryless': h_memoryless,
                    'h_optimal': h_optimal,
                })
                print(f"BPS={sm_bps:.4f}  EvRecall={sm_acc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(results):
    w = 90
    header = (f"{'k':>3} {'Model':>6} {'EM':>3} {'C':>3} {'L':>3} "
              f"{'BPS':>8} {'Optim':>8} {'Base':>8} "
              f"{'EvRecall':>8} {'KL':>8} {'Iters':>5}")
    print(f"\n{'='*w}")
    print("RESULTS  (Optim = theoretical optimal BPS, Base = memoryless entropy)")
    print("EvRecall = fraction of actual 1s correctly predicted as 1")
    print("=" * w)
    print(header)
    print("-" * w)
    for r in results:
        kl_str = (f"{r['kl_divergence']:>8.4f}"
                  if not np.isnan(r['kl_divergence']) else f"{'—':>8}")
        print(f"{str(r['k']):>3} {r['model']:>6} {r['em']:>3} "
              f"{r['n_clones']:>3} {r['n_phases']:>3} "
              f"{r['test_bps']:>8.4f} {r['h_optimal']:>8.4f} "
              f"{r['h_memoryless']:>8.4f} "
              f"{r['accuracy']:>8.4f} {kl_str} {r['n_iters']:>5}")


def plot_results(results, save_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plots)")
        return

    import pandas as pd
    df = pd.DataFrame(results)
    ks = sorted(df['k'].unique(), key=lambda x: (0, x) if isinstance(x, int) else (1, x))

    # Style map: (model, em) → (marker, linestyle)
    variants = [
        ('CSCG',   'BW',  'o', '-'),
        ('CSCG',   'Vit', 'o', '--'),
        ('SMCSCG', 'BW',  's', '-'),
        ('SMCSCG', 'Vit', 's', '--'),
    ]
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for k_idx, k in enumerate(ks):
        for v_idx, (model, em, marker, ls) in enumerate(variants):
            sub = df[(df['k'] == k) & (df['model'] == model) & (df['em'] == em)]
            if sub.empty:
                continue
            color = colors[(k_idx * len(variants) + v_idx) % len(colors)]
            label = f"k={k} {model} {em}"
            axes[0].plot(sub['n_clones'], sub['test_bps'],
                         marker=marker, linestyle=ls, color=color, label=label)
            axes[1].plot(sub['n_clones'], sub['accuracy'],
                         marker=marker, linestyle=ls, color=color, label=label)

    # Add theoretical optimal BPS lines
    for k_idx, k in enumerate(ks):
        sub_k = df[df['k'] == k]
        if sub_k.empty:
            continue
        h_opt = sub_k['h_optimal'].iloc[0]
        h_mem = sub_k['h_memoryless'].iloc[0]
        axes[0].axhline(h_opt, color='grey', linestyle=':', alpha=0.6,
                         label=f'k={k} optimal' if k_idx == 0 else None)
        axes[0].axhline(h_mem, color='grey', linestyle='--', alpha=0.4,
                         label=f'k={k} memoryless' if k_idx == 0 else None)

    axes[0].set(xlabel='n_clones', ylabel='BPS',
                title='Bits per Symbol (lower = better)')
    axes[1].set(xlabel='n_clones', ylabel='Event Recall',
                title='Event Recall (1s predicted as 1)')

    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n  Plot saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Erlang-k duration experiment: CSCG vs SM-CSCG"
    )
    parser.add_argument("--platform", default="cpu",
                        choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--erlang_shapes", type=int, nargs="+",
                        default=[2, 4, 6])
    parser.add_argument("--clone_counts", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--mean_gap", type=float, default=10,
                        help="Mean inter-event gap (p derived per k as k/(mean_gap+k))")
    parser.add_argument("--seq_length", type=int, default=5000)
    parser.add_argument("--n_train", type=int, default=20)
    parser.add_argument("--n_test", type=int, default=5)
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--sample_length", type=int, default=5000)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", type=str, default=None,
                        help="Save plot to path (e.g., results/erlang.png)")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test with reduced parameters")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", args.platform)

    if args.quick:
        args.erlang_shapes = [2, 4]
        args.clone_counts = [1, 2, 4]
        args.seq_length = 3000
        args.n_train = 10
        args.n_test = 3
        args.n_sample = 5
        args.sample_length = 3000
        args.n_iter = 50

    print("Erlang-k Duration Experiment: CSCG vs SM-CSCG")
    print(f"  Erlang shapes: {args.erlang_shapes}")
    print(f"  Clone counts:  {args.clone_counts}")
    print(f"  SM-CSCG n_phases = k (matched to Erlang shape)")
    print(f"  EM methods: Baum-Welch + Viterbi")
    print(f"  mean_gap={args.mean_gap}, seq_length={args.seq_length}")

    results = run_experiment(args)
    print_results_table(results)

    if args.plot:
        plot_results(results, args.plot)


if __name__ == "__main__":
    main()
