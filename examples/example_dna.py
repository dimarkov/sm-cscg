#!/usr/bin/env python3
"""Example 3: DNA sequence compression benchmark.

Compares CSCG vs SM-CSCG on genomic DNA (E. coli K-12 fragment).
DNA has alphabet size 4 (A=0, C=1, G=2, T=3) and features:
  - Homopolymer runs with non-geometric length distributions
  - Dinucleotide context patterns (CpG suppression, etc.)
  - Codon-scale periodicities in coding regions

SM-CSCG's phase-type duration modeling should capture homopolymer
run distributions better than CSCG's implicit geometric durations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import jax
from smcscg import CSCG, SMCSCG, metrics


N_OBS = 4
NUC_NAMES = {0: "A", 1: "C", 2: "G", 3: "T"}

# E. coli K-12 substr. MG1655 genomic DNA, positions 190000-194500
# (gene-rich region with typical ~50% GC content)
# Source: Ensembl, accession U00096.3
DNA_SEQUENCE = (
    "CTTGAGAAAACTGTACCGATGTTCAACGAAGCTCTGGCTGAACTGAACAAGATTGCTTCTCGCAAAGGT"
    "AAAATCCTTTTCGTTGGTACTAAACGCGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGCTGCGAC"
    "CAGTTCTTCGTGAACCATCGCTGGCTGGGCGGTATGCTGACTAACTGGAAAACCGTTCGTCAGTCCATCA"
    "AACGTCTGAAAGACCTGGAAACTCAGTCTCAGGACGGTACTTTCGACAAGCTGACCAAGAAAGAAGCGCT"
    "GATGCGCACTCGTGAGCTGGAGAAACTGGAAAACAGCCTGGGCGGTATCAAAGACATGGGCGGTCTGCCG"
    "GACGCTCTGTTTGTAATCGATGCTGACCACGAACACATTGCTATCAAAGAAGCAAACAACCTGGGTATTC"
    "CGGTATTTGCTATCGTTGATACCAACTCTGATCCGGACGGTGTTGACTTCGTTATCCCGGGTAACGACGA"
    "CGCAATCCGTGCTGTGACCCTGTACCTGGGCGCTGTTGCTGCAACCGTACGTGAAGGCCGTTCTCAGGAT"
    "CTGGCTTCCCAGGCGGAAGAAAGCTTCGTAGAAGCTGAGTAATAAGGCTTGATAACTCCCCCAAAATAGTC"
    "CGAGTTGCAGAAAGGCGGCAAGCTCGAGAATTCCCGGGAGCTTACATCAGTAAGTGACCGGGATGAGCGA"
    "GCGAAGATAACGCATCTGCGGCGCGAAATATGAAGGGGGAGAGCCCTTATAGACCAGGTAGTACACGTTTG"
    "GTTAGGGGGCCTGCATATGGCCCCCTTTTTCACTTTTATATCTGTGCGGTTTAATGCCGGGCAGATCACAT"
    "CTCCGAGGATTTTAGAATGGCTGAAATTACCGCATCCCTGGTAAAAGAGCTGCGTGAGCGTACTGGCGCAG"
    "GCATGATGGATTGCAAAAAAGCACTGACTGAAGCTAACGGCGACATCGAGCTGGCAATCGAAAACATGCGT"
    "AAGTCCGGTGCTATTAAAGCAGCGAAAAAAGCAGGCAACGTTGCTGCTGACGGCGTGATCAAAACCAAAAT"
    "CGACGGCAACTACGGCATCATTCTGGAAGTTAACTGCCAGACTGACTTCGTTGCAAAAGACGCTGGTTTCC"
    "AGGCGTTCGCAGACAAAGTTCTGGACGCAGCTGTTGCTGGCAAAATCACTGACGTTGAAGTTCTGAAAGCA"
    "CAGTTCGAAGAAGAACGTGTTGCGCTGGTAGCGAAAATTGGTGAAAACATCAACATTCGCCGCGTTGCTGC"
    "GCTGGAAGGCGACGTTCTGGGTTCTTATCAGCACGGTGCGCGTATCGGCGTTCTGGTTGCTGCTAAAGGCG"
    "CTGACGAAGAGCTGGTTAAACACATCGCTATGCACGTTGCTGCAAGCAAGCCAGAATTCATCAAACCGGAAG"
    "ACGTATCCGCTGAAGTGGTAGAAAAAGAATACCAGGTACAGCTGGATATCGCGATGCAGTCTGGTAAGCCGA"
    "AAGAAATCGCAGAGAAAATGGTTGAAGGCCGCATGAAGAAATTCACCGGCGAAGTTTCTCTGACCGGTCAGC"
    "CGTTCGTTATGGAACCAAGCAAAACTGTTGGTCAGCTGCTGAAAGAGCATAACGCTGAAGTGACTGGCTTCA"
    "TCCGCTTCGAAGTGGGTGAAGGCATCGAGAAAGTTGAGACTGACTTTGCAGCAGAAGTTGCTGCGATGTCCA"
    "AGCAGTCTTAATTATCAAAAAGGAGCCGCCTGAGGGCGGCTTCTTTTTGTGCCCATCTTGTAAATTCAGCTAA"
    "CCCTTGTGGGGCTGCGCTGAAAAGCGACGTACAATGTCGCTAGTATTAATTCATTTCAATCGTTGACAGTCTC"
    "AGGAAAGAAACATGGCTACCAATGCAAAACCCGTCTATAAACGCATTCTGCTTAAGTTGAGTGGCGAAGCTCTG"
    "CAGGGCACTGAAGGCTTCGGTATTGATGCAAGCATACTGGATCGTATGGCTCAGGAAATCAAAGAACTGGTTGA"
    "ACTGGGTATTCAGGTTGGTGTGGTGATTGGTGGGGGTAACCTGTTCCGTGGCGCTGGTCTGGCGAAAGCGGGT"
    "ATGAACCGCGTTGTGGGCGACCACATGGGGATGCTGGCGACCGTAATGAACGGCCTGGCAATGCGTGATGCAC"
    "TGCACCGCGCCTATGTGAACGCTCGTCTGATGTCCGCTATTCCATTGAATGGCGTGTGCGACAGCTACAGCTGG"
    "GCAGAAGCTATCAGCCTGTTGCGCAACAACCGTGTGGTGATCCTCTCCGCCGGTACAGGTAACCCGTTCTTTAC"
    "CACCGACTCAGCAGCTTGCCTGCGTGGTATCGAAATTGAAGCCGATGTGGTGCTGAAAGCAACCAAAGTTGACG"
    "GCGTGTTTACCGCTGATCCGGCGAAAGATCCAACCGCAACCATGTACGAGCAACTGACTTACAGCGAAGTGCTGG"
    "AAAAAGAGCTGAAAGTCATGGACCTGGCGGCCTTCACGCTGGCTCGTGACCATAAATTACCGATTCGTGTTTTCAA"
    "TATGAACAAACCGGGTGCGCTGCGCCGTGTGGTAATGGGTGAAAAAGAAGGGACTTTAATCACGGAATAATTCCCG"
    "TGATGGATAAATAAGGGTAAGATTCCGCGTAAGTATCGCGGGGGCGTAAGTCTGGTTATAAGGCGTTATTGTTGCA"
    "GGCAGTTTGGTCACGGCCAGCGCGCAGCAACCGGAGCGTACAAAAGTACGTGAGGATGGCGAGCACTGCCCGGGGC"
    "CAAAATGGCAAATAAAATAGCCTAATAATCCAGACGATTACCCGTAATATGTTTAATCAGGGCTATACTTAGCACACT"
    "TCCACTGTGTGTGACTGTCTGGTCTGACTGAGACAAGTTTTCAAGGATTCGTAACGTGATTAGCGATATCAGAAAAG"
    "ATGCTGAAGTACGCATGGACAAATGCGTAGAAGCGTTCAAAACCCAAATCAGCAAAATACGCACGGGTCGTGCTTCT"
    "CCCAGCCTGCTGGATGGCATTGTCGTGGAATATTACGGCACGCCGACGCCGCTGCGTCAGCTGGCAAGCGTAACGG"
    "TAGAAGATTCCCGTACACTGAAAATCAACGTGTTTGATCGTTCAATGTCTCCGGCCGTTGAAAAAGCGATTATGGCG"
    "TCCGATCTTGGCCTGAACCCGAACTCTGCGGGTAGCGACATCCGTGTTCCGCTGCCGCCGCTGACGGAAGAACGTCG"
    "TAAAGATCTGACCAAAATCGTTCGTGGTGAAGCAGAACAAGCGCGTGTTGCAGTACGTAACGTGCGTCGTGACGCGA"
    "ACGACAAAGTGAAAGCACTGTTGAAAGATAAAGAGATCAGCGAAGACGACGATCGCCGTTCTCAGGACGATGTACAGA"
    "AACTGACTGATGCTGCAATCAAGAAAATTGAAGCGGCGCTGGCAGACAAAGAAGCAGAACTGATGCAGTTCTGATTTC"
    "TTGAACGACAAAAACGCCGCTCAGTAGATCCTTGCGGATCGGCTGGCGGCGTTTTGCTTTTTATTCTGTCTCAACTCTG"
    "GATGTTTCATGAAGCAACTCACCATTCTGGGCTCGACCGGCTCGATTGGTTGCAGCACGCTGGACGTGGTGCGCCATAA"
    "TCCCGAACACTTCCGCGTAGTTGCGCTGGTGGCAGGCAAAAATGTCACTCGCATGGTAGAACAGTGCCTGGAATTCTCTC"
    "CCCGCTATGCCGTAATGGACGATGAAGCGAGTGCGAAACTTCTTAAAACGATGCTACAGCAACAGGGTAGCCGCACCGAA"
    "GTCTTAAGTGGGCAACAAGCCGCTTGCGATATGGCAGCGCTTGAGGATGTTGATCAGGTGATGGCAGCCATTGTTGGCGC"
    "TGCTGGGCTGTTACCTACGCTTGCTGCGATCCGCGCGGGTAAAACCATTTTGCTGGCCAATAAAGAATCACTGGTTACCTG"
    "CGGACGTCTGTTTATGGACGCCGTAAAGCAGAGCAAAGCGCAATTGTTACCGGTCGATAGCGAACATAACGCCATTTTTCA"
    "GAGTTTACCGCAACCTATCCAGCATAATCTGGGATACGCTGACCTTGAGCAAAATGGCGTGGTGTCCATTTTACTTACCGGG"
    "TCTGGTGGCCCTTTCCGTGAGACGCCATTGCGCGATTTGGCAACAATGACGCCGGATCAAGCCTGCCGTCATCCGAACTGG"
    "TCGATGGGGCGTAAAATTTCTGTCGATTCGGCTACCATGATGAACAAAGGTCTGGAATACATTGAAGCGCGTTGGCTGTTTA"
    "ACGCCAGCGCCAGCCAGATGGAAGTGCTGATTCACCCGCAGTCAGTGATTCACTCAATGGTGCGCTATCAGGACGGCAGTGT"
    "TCTGGCGCAGCTGGGGGAACCGGATATGCGTACGCCAATTGCCCACACCATGGCATGGCCGAATCGCGTGAACTCTGGCGTGA"
    "AGCCGCTCGATTTTTGCAAACTAAGTGCGTTGACATTTGCCGCACCGGATTATGATCGTTATCCATGCCTGAAACTGGCGATGG"
    "AGGCGTTCGAACA"
)


def encode_dna(seq_str):
    """Convert DNA string to integer observation sequence."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    return np.array(
        [mapping[c] for c in seq_str.upper() if c in mapping], dtype=np.intp
    )


def split_into_chunks(obs_seq, chunk_size=200):
    """Split observation sequence into chunks."""
    chunks = []
    for i in range(0, len(obs_seq), chunk_size):
        chunk = obs_seq[i:i + chunk_size]
        if len(chunk) >= 40:
            chunks.append(chunk)
    return chunks


def compute_homopolymer_stats(obs_seq):
    """Compute run-length distributions for each nucleotide.

    Returns
    -------
    run_lengths : dict mapping obs_idx -> list of run lengths
    """
    run_lengths = {o: [] for o in range(N_OBS)}
    current_sym = obs_seq[0]
    current_len = 1
    for t in range(1, len(obs_seq)):
        if obs_seq[t] == current_sym:
            current_len += 1
        else:
            run_lengths[current_sym].append(current_len)
            current_sym = obs_seq[t]
            current_len = 1
    run_lengths[current_sym].append(current_len)
    return run_lengths


def compute_baselines(obs_seq):
    """Compute information-theoretic BPS baselines.

    Returns
    -------
    uniform_bps, unigram_bps, bigram_bps : float
    """
    # Uniform: log2(4) = 2.0
    uniform_bps = np.log2(N_OBS)

    # Unigram entropy
    counts = np.bincount(obs_seq, minlength=N_OBS)
    freqs = counts / counts.sum()
    freqs_nz = freqs[freqs > 0]
    unigram_bps = -np.sum(freqs_nz * np.log2(freqs_nz))

    # Bigram conditional entropy: H(X_{t+1} | X_t)
    bigram_counts = np.zeros((N_OBS, N_OBS))
    for t in range(len(obs_seq) - 1):
        bigram_counts[obs_seq[t], obs_seq[t + 1]] += 1
    row_sums = bigram_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bigram_probs = bigram_counts / row_sums
    marginal = counts / counts.sum()
    bigram_bps = 0.0
    for i in range(N_OBS):
        for j in range(N_OBS):
            if bigram_probs[i, j] > 0:
                bigram_bps -= marginal[i] * bigram_probs[i, j] * np.log2(bigram_probs[i, j])

    return uniform_bps, unigram_bps, bigram_bps


def analyze_homopolymer_durations(smcscg, obs_seq):
    """Compare learned phase-type durations to empirical homopolymer runs."""
    empirical = compute_homopolymer_stats(obs_seq)

    print("\n  Learned vs empirical homopolymer durations:")
    for o in range(N_OBS):
        runs = empirical[o]
        emp_mean = np.mean(runs) if runs else 0
        emp_max = max(runs) if runs else 0
        frac_ge3 = sum(1 for r in runs if r >= 3) / max(len(runs), 1)

        # Find clone with longest and shortest learned mean duration
        macro_clones = list(range(o * smcscg.n_clones, (o + 1) * smcscg.n_clones))
        clone_means = []
        for j in macro_clones:
            pmf = smcscg.duration_pmf(j, max_d=15)
            ds = np.arange(1, len(pmf) + 1)
            total = max(pmf.sum(), 1e-12)
            clone_means.append((j, np.dot(ds, pmf) / total))

        clone_means.sort(key=lambda x: -x[1])
        best_j, best_mean = clone_means[0]
        short_j, short_mean = clone_means[-1]

        print(f"    {NUC_NAMES[o]}: empirical mean={emp_mean:.2f} max={emp_max} "
              f"(>= 3: {frac_ge3:.0%})")
        print(f"       longest clone {best_j}: mean={best_mean:.2f}, "
              f"shortest clone {short_j}: mean={short_mean:.2f}")


def analyze_clone_context(model, sequences, obs_idx):
    """Analyze trinucleotide contexts for different clones of a nucleotide."""
    clone_contexts = {}

    for seq in sequences[:10]:
        states = model.decode(seq)
        if isinstance(states, tuple):
            states = states[0]
        for t in range(len(seq)):
            if seq[t] == obs_idx:
                clone_id = int(states[t])
                prev = NUC_NAMES.get(int(seq[t - 1]), "^") if t > 0 else "^"
                nxt = NUC_NAMES.get(int(seq[t + 1]), "$") if t < len(seq) - 1 else "$"
                ctx = prev + NUC_NAMES[obs_idx] + nxt
                if clone_id not in clone_contexts:
                    clone_contexts[clone_id] = {}
                clone_contexts[clone_id][ctx] = clone_contexts[clone_id].get(ctx, 0) + 1

    print(f"\n  Clone specialization for {NUC_NAMES[obs_idx]}:")
    for cid, contexts in sorted(clone_contexts.items()):
        top = sorted(contexts.items(), key=lambda x: -x[1])[:4]
        top_str = ", ".join(f"{c}:{n}" for c, n in top)
        total = sum(contexts.values())
        print(f"    Clone {cid} ({total} occ): {top_str}")


def run_example():
    print("=" * 60)
    print("DNA sequence compression benchmark")
    print("=" * 60)

    obs_seq = encode_dna(DNA_SEQUENCE)
    print(f"\n  Sequence length: {len(obs_seq)} nucleotides")

    # Base composition
    counts = np.bincount(obs_seq, minlength=N_OBS)
    freqs = counts / counts.sum()
    comp = ", ".join(f"{NUC_NAMES[o]}={freqs[o]:.1%}" for o in range(N_OBS))
    print(f"  Base composition: {comp}")

    # Homopolymer stats
    hp = compute_homopolymer_stats(obs_seq)
    print("\n  Homopolymer run-length statistics:")
    for o in range(N_OBS):
        runs = hp[o]
        if runs:
            mean_r = np.mean(runs)
            max_r = max(runs)
            frac3 = sum(1 for r in runs if r >= 3) / len(runs)
            print(f"    {NUC_NAMES[o]}: mean={mean_r:.2f}, max={max_r}, "
                  f"runs >= 3: {frac3:.0%}")

    # Baselines
    uni, ugram, bigram = compute_baselines(obs_seq)
    print(f"\n  Information-theoretic baselines:")
    print(f"    Uniform:  {uni:.4f} BPS")
    print(f"    Unigram:  {ugram:.4f} BPS")
    print(f"    Bigram:   {bigram:.4f} BPS")

    # Split
    chunks = split_into_chunks(obs_seq, chunk_size=200)
    n_train = max(1, int(len(chunks) * 0.75))
    train_chunks = chunks[:n_train]
    test_chunks = chunks[n_train:]
    print(f"\n  Train chunks: {len(train_chunks)}, Test chunks: {len(test_chunks)} "
          f"(chunk_size=200)")

    # Model configs: (label, model_type, em_method, kwargs)
    configs = [
        ("CSCG K=10 BW",      "cscg",   "baum-welch", {"n_clones": 10}),
        ("CSCG K=10 Vit",     "cscg",   "viterbi",    {"n_clones": 10}),
        ("CSCG K=20 BW",      "cscg",   "baum-welch", {"n_clones": 20}),
        ("CSCG K=20 Vit",     "cscg",   "viterbi",    {"n_clones": 20}),
        ("SM-CSCG K=5 L=4 BW",  "smcscg", "baum-welch", {"n_clones": 5,  "n_phases": 4}),
        ("SM-CSCG K=5 L=4 Vit", "smcscg", "viterbi",    {"n_clones": 5,  "n_phases": 4}),
        ("SM-CSCG K=10 L=4 BW", "smcscg", "baum-welch", {"n_clones": 10, "n_phases": 4}),
        ("SM-CSCG K=10 L=4 Vit","smcscg", "viterbi",    {"n_clones": 10, "n_phases": 4}),
    ]

    results = []
    best_smcscg = None

    for label, mtype, em_method, kwargs in configs:
        n_clones = kwargs["n_clones"]
        print(f"\n  --- {label} ---")

        if mtype == "cscg":
            model, lls = CSCG(n_obs=N_OBS, n_clones=n_clones,
                               key=jax.random.PRNGKey(42)).fit(
                train_chunks, n_iter=30, verbose=False, em_method=em_method
            )
            model_name = "CSCG"
        else:
            n_phases = kwargs["n_phases"]
            model, lls = SMCSCG(n_obs=N_OBS, n_clones=n_clones,
                                 n_phases=n_phases, phase_type="coxian",
                                 key=jax.random.PRNGKey(42)).fit(
                train_chunks, n_iter=20, verbose=False, em_method=em_method
            )
            model_name = "SM-CSCG"
            if em_method == "baum-welch":
                best_smcscg = model

        train_bps = metrics.bits_per_symbol(model, train_chunks)
        test_bps = metrics.bits_per_symbol(model, test_chunks)
        print(f"    Train BPS: {train_bps:.4f}, Test BPS: {test_bps:.4f} "
              f"({len(lls)} iters)")

        results.append({
            "model": f"{model_name} ({em_method[:3]})", "dataset": "dna",
            "n_clones": n_clones,
            "train_bps": train_bps, "test_bps": test_bps,
        })

    # --- DNA-specific analysis ---

    # Homopolymer duration analysis (best SM-CSCG)
    if best_smcscg is not None:
        analyze_homopolymer_durations(best_smcscg, obs_seq)

        # Clone context specialization for G (most interesting due to CpG)
        analyze_clone_context(best_smcscg, test_chunks, obs_idx=2)  # G

    # Summary
    print("\n" + "=" * 60)
    print("DNA COMPRESSION SUMMARY")
    print("=" * 60)
    print(f"  Baselines: uniform={uni:.4f}, unigram={ugram:.4f}, "
          f"bigram={bigram:.4f} BPS")
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
