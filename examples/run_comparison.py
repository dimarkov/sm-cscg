#!/usr/bin/env python3
"""Unified comparison: run both examples and produce summary tables."""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
from smcscg import metrics
from examples.example_symbols import run_no_duration, run_with_duration
from examples.example_chartext import run_example as run_chartext
from examples.example_dna import run_example as run_dna


def main():
    print("=" * 70)
    print("  CSCG vs SM-CSCG: Unified Comparison")
    print("=" * 70)

    all_results = []

    # Symbol patterns
    r1 = run_no_duration()
    all_results.append(r1)

    r2_list = run_with_duration()
    all_results.extend(r2_list)

    # Character text
    r3_list = run_chartext()
    all_results.extend(r3_list)

    # DNA sequences
    r4_list = run_dna()
    all_results.extend(r4_list)

    # Final unified table
    print("\n" + "=" * 70)
    print("  UNIFIED COMPARISON TABLE")
    print("=" * 70)
    metrics.print_comparison_table(all_results)

    print("\n  Key observations:")
    print("  - CSCG resolves observation aliasing via clone specialization")
    print("  - SM-CSCG additionally models segment durations via phase-type distributions")
    print("  - On duration data, SM-CSCG achieves better (lower) BPS")
    print("  - SM-CSCG recovers segment boundaries with high F1")
    print("  - On character text, SM-CSCG captures word-like segments")
    print("  - On DNA, SM-CSCG captures homopolymer run-length distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="cpu",
                        choices=["cpu", "gpu", "tpu"],
                        help="JAX platform (default: cpu)")
    args = parser.parse_args()
    jax.config.update("jax_platform_name", args.platform)
    main()
