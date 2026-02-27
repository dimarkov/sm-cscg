#!/usr/bin/env python3
"""Example 2: Character-level text encoding.

Encodes a small text corpus at the character level.
CSCG learns context-dependent clone specialization for repeated characters.
SM-CSCG additionally discovers segment structure (word/syllable boundaries).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import jax
from smcscg import CSCG, SMCSCG, metrics


# Public-domain text (opening of "A Tale of Two Cities" by Dickens)
CORPUS = (
    "it was the best of times it was the worst of times "
    "it was the age of wisdom it was the age of foolishness "
    "it was the epoch of belief it was the epoch of incredulity "
    "it was the season of light it was the season of darkness "
    "it was the spring of hope it was the winter of despair "
    "we had everything before us we had nothing before us "
    "we were all going direct to heaven "
    "we were all going direct the other way "
    "in short the period was so far like the present period "
    "that some of its noisiest authorities insisted on its "
    "being received for good or for evil in the superlative "
    "degree of comparison only "
)


def encode_text(text):
    """Convert text to integer observation sequence.

    Returns
    -------
    obs_seq : ndarray of int
    char_to_idx : dict
    idx_to_char : dict
    """
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    obs_seq = np.array([char_to_idx[c] for c in text], dtype=np.intp)
    return obs_seq, char_to_idx, idx_to_char


def split_into_chunks(obs_seq, chunk_size=100):
    """Split an observation sequence into chunks."""
    chunks = []
    for i in range(0, len(obs_seq), chunk_size):
        chunk = obs_seq[i:i + chunk_size]
        if len(chunk) >= 20:  # skip very short trailing chunks
            chunks.append(chunk)
    return chunks


def visualize_segmentation(text_chunk, segments, idx_to_char):
    """Show segmentation overlaid on text."""
    result = []
    for state, start, dur in segments:
        segment_text = "".join(idx_to_char[int(c)] for c in text_chunk[start:start + dur])
        result.append(f"[{segment_text}]")
    return "".join(result)


def run_example():
    print("=" * 60)
    print("Character-level text encoding")
    print("=" * 60)

    obs_seq, char_to_idx, idx_to_char = encode_text(CORPUS)
    n_obs = len(char_to_idx)

    print(f"\n  Corpus length: {len(obs_seq)} characters")
    print(f"  Alphabet size: {n_obs} unique characters")
    print(f"  Characters: {''.join(sorted(char_to_idx.keys()))}")

    # Split into train/test
    chunks = split_into_chunks(obs_seq, chunk_size=80)
    n_train = max(1, int(len(chunks) * 0.75))
    train_chunks = chunks[:n_train]
    test_chunks = chunks[n_train:]

    print(f"  Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)}")

    results = []

    for n_clones in [5, 10]:
        print(f"\n  --- n_clones = {n_clones} ---")

        # CSCG
        print(f"  Training CSCG...")
        cscg, lls = CSCG(n_obs=n_obs, n_clones=n_clones, key=jax.random.PRNGKey(42)).fit(
            train_chunks, n_iter=25, verbose=False
        )

        train_bps = metrics.bits_per_symbol(cscg, train_chunks)
        test_bps = metrics.bits_per_symbol(cscg, test_chunks)
        print(f"    CSCG Train BPS: {train_bps:.4f}, Test BPS: {test_bps:.4f}")

        results.append({
            "model": "CSCG", "dataset": "char-text",
            "n_clones": n_clones,
            "train_bps": train_bps, "test_bps": test_bps,
        })

        # SM-CSCG
        print(f"  Training SM-CSCG (n_phases=5, coxian)...")
        smcscg, lls = SMCSCG(n_obs=n_obs, n_clones=n_clones, n_phases=5,
                              phase_type="coxian", key=jax.random.PRNGKey(42)).fit(
            train_chunks, n_iter=15, verbose=False
        )

        train_bps_sm = metrics.bits_per_symbol(smcscg, train_chunks)
        test_bps_sm = metrics.bits_per_symbol(smcscg, test_chunks)
        print(f"    SM-CSCG Train BPS: {train_bps_sm:.4f}, Test BPS: {test_bps_sm:.4f}")

        results.append({
            "model": "SM-CSCG", "dataset": "char-text",
            "n_clones": n_clones,
            "train_bps": train_bps_sm, "test_bps": test_bps_sm,
        })

        # Visualize SM-CSCG segmentation on a test chunk
        if test_chunks and n_clones == 5:
            print(f"\n  SM-CSCG segmentation of test chunk:")
            chunk = test_chunks[0]
            _, segments = smcscg.decode(chunk)
            viz = visualize_segmentation(chunk, segments, idx_to_char)
            print(f"    {viz}")

    # Summary
    print("\n" + "=" * 60)
    print("CHARACTER TEXT SUMMARY")
    print("=" * 60)
    metrics.print_comparison_table(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="cpu",
                        choices=["cpu", "gpu", "tpu"],
                        help="JAX platform (default: cpu)")
    args = parser.parse_args()
    jax.config.update("jax_platform_name", args.platform)
    run_example()
