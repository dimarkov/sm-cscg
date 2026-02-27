"""Comparison metrics for CSCG vs SM-CSCG models."""

import numpy as np


def total_log_likelihood(model, sequences):
    """Sum of log-likelihoods across all sequences."""
    return sum(model.log_likelihood(seq) for seq in sequences)


def per_symbol_log_likelihood(model, sequences):
    """Average log-likelihood per symbol across all sequences."""
    total_ll = 0.0
    total_len = 0
    for seq in sequences:
        total_ll += model.log_likelihood(seq)
        total_len += len(seq)
    return total_ll / total_len if total_len > 0 else 0.0


def bits_per_symbol(model, sequences):
    """Average bits per symbol = -per_symbol_ll / ln(2)."""
    psll = per_symbol_log_likelihood(model, sequences)
    return -psll / np.log(2)


def segment_boundaries(segments):
    """Extract boundary positions from a list of (state, start, duration) tuples.

    Returns a sorted set of start positions (excluding 0).
    """
    return sorted(s[1] for s in segments if s[1] > 0)


def segment_f1(predicted_segments, ground_truth_segments, tolerance=1):
    """Precision, recall, F1 for segment boundary recovery.

    A predicted boundary is correct if it falls within `tolerance`
    positions of a ground-truth boundary.

    Parameters
    ----------
    predicted_segments : list of (state, start, duration)
    ground_truth_segments : list of (state, start, duration)
    tolerance : int

    Returns
    -------
    precision, recall, f1 : float
    """
    pred_bounds = segment_boundaries(predicted_segments)
    true_bounds = segment_boundaries(ground_truth_segments)

    if not pred_bounds and not true_bounds:
        return 1.0, 1.0, 1.0
    if not pred_bounds or not true_bounds:
        return 0.0, 0.0, 0.0

    # Match predicted to true
    true_matched = set()
    tp = 0
    for pb in pred_bounds:
        for i, tb in enumerate(true_bounds):
            if i not in true_matched and abs(pb - tb) <= tolerance:
                tp += 1
                true_matched.add(i)
                break

    precision = tp / len(pred_bounds) if pred_bounds else 0.0
    recall = tp / len(true_bounds) if true_bounds else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def print_comparison_table(results):
    """Print a formatted comparison table.

    Parameters
    ----------
    results : list of dict
        Each dict has keys: 'model', 'dataset', 'n_clones', 'train_bps',
        'test_bps', and optionally 'seg_f1'.
    """
    header = f"{'Model':<12} {'Dataset':<16} {'Clones':>6} {'Train BPS':>10} {'Test BPS':>10} {'Seg F1':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        seg_f1_str = f"{r.get('seg_f1', float('nan')):.3f}" if 'seg_f1' in r else "   N/A"
        print(f"{r['model']:<12} {r['dataset']:<16} {r['n_clones']:>6} "
              f"{r['train_bps']:>10.4f} {r['test_bps']:>10.4f} {seg_f1_str:>8}")
