# Phase 2 Plan

## Context

Phase 1 delivered:
- JAX/Equinox CSCG and SM-CSCG with scan-vectorised Baum-Welch and Viterbi EM
- Coxian and general phase-type duration distributions
- GINC in-context learning experiment (replicating Swaminathan et al. NeurIPS 2023, CSCG only)
- Examples: symbol patterns, character text, DNA compression

The central hypothesis motivating this project is that **SM-CSCG** — which combines
clone-based structural disambiguation (CSCG) with explicit segment-duration modeling
(phase-type distributions) — should outperform plain CSCG in two complementary
ways:

1. **Compression**: better BPS on data with non-geometric run-length distributions
   (homopolymer runs, word/syllable durations, workload bursts)
2. **In-context learning**: faster schema rebinding when examples have variable
   segment lengths, because the model can infer "how long" a segment should run
   from fewer examples

---

## Goal 1: SM-CSCG on the GINC Benchmark

**Motivation.** The GINC data generator produces sentences of fixed length `k`
drawn from a latent HMM schema. Each "sentence" is a segment of identical
length. Phase-type durations add no value here — which is probably why the
original paper only uses CSCG. The interesting experiment is a **variable-length
GINC variant** where sentence lengths are drawn from a non-geometric distribution
(e.g., Poisson or Erlang). In this setting SM-CSCG should learn the duration
distribution and rebind schemas with fewer context examples.

**Tasks:**

1. Add a `--duration_dist` flag to `experiments/ginc_data.py` and
   `experiments/generate_data.py`:
   - `fixed` (default, current behaviour)
   - `geometric` (sanity check: SM-CSCG ≈ CSCG)
   - `poisson` / `erlang` (SM-CSCG should win)

2. Add SM-CSCG training path to `experiments/ginc_experiment.py`:
   - `--model {cscg, smcscg}` flag
   - When `smcscg`: choose `n_phases`, `phase_type`
   - Report ICL accuracy table with the same (k, n) grid

3. Compare CSCG vs SM-CSCG accuracy curves as a function of `n` (number of
   context examples) for each duration distribution.

**Expected outcome.** On Poisson/Erlang durations: SM-CSCG reaches target
accuracy (~97% at k=8, n=8) with fewer context examples and/or smaller n_clones.

---

## Goal 2: Real Sequence Compression Benchmarks

**Motivation.** The DNA and chartext examples show BPS numbers but do not
compare against established baselines (arithmetic coding, PPM, LZ). Adding
proper baselines turns the examples into reproducible compression benchmarks.

**Tasks:**

1. Add a `benchmarks/` module with:
   - `compress_zstd(data)` — via the `zstandard` package
   - `compress_bz2(data)` — stdlib
   - `compress_lzma(data)` — stdlib (approximates PPM)
   - Each returns BPS on the byte-encoded sequence

2. Extend `example_dna.py` and `example_chartext.py` to print a unified table
   including classical compressor BPS alongside CSCG / SM-CSCG BPS.

3. Add a longer DNA benchmark (full E. coli chromosome or a 1 MB FASTA chunk)
   via a `data/` download script.

**Expected outcome.** SM-CSCG beats CSCG on homopolymer-rich sequences; both
beat unigram/bigram but are below zstd on raw bytes (as expected for a small
HMM). The table provides honest context for what the model achieves.

---

## Goal 3: Dynamic Clone Count (Variable K per Observation)

**Motivation.** The current implementation uses a fixed `n_clones` K for all
observations. In practice, frequent/ambiguous observations need more clones;
rare/unambiguous ones need fewer. The DSMM literature suggests allocating clones
adaptively based on usage statistics.

**Tasks:**

1. Allow `n_clones` to be a list/array of length `n_obs` (one K per observation).
   Update `_numerics.py` (`build_clone_emission_matrix`, `clones_for_obs`) and the
   `CSCG` / `SMCSCG` constructors.

2. Add a `split_clone(obs_idx, clone_idx)` method to CSCG that:
   - Duplicates the specified clone (copies its row/column in `log_T`)
   - Splits incoming transition mass proportionally
   - Returns a new model with one extra clone for that observation

3. Add a `prune_clone(obs_idx, clone_idx)` method that merges an underused clone
   back into another.

4. Add a `fit_with_splitting(sequences, ...)` training loop:
   - Run EM to convergence at current K
   - Compute per-clone usage (marginal posterior mass)
   - Split clones above a usage threshold; prune below a minimum threshold
   - Repeat until K stabilises or a max-clones budget is reached

**Expected outcome.** Adaptive K reduces parameter count vs fixed K while
matching or improving BPS; clone specialization becomes cleaner.

---

## Goal 4: Mini-batch / Online EM

**Motivation.** All current training loads all sequences into a single padded
batch. For long corpora (full chromosome, GINC with 10 k documents) this exceeds
GPU memory. Mini-batch EM accumulates sufficient statistics over sub-batches and
applies one M-step per pass.

**Tasks:**

1. Add `fit_minibatch(sequences, batch_size, n_epochs, ...)` to both CSCG and
   SMCSCG:
   - Shuffle sequences each epoch
   - For each mini-batch: run `_e_step_scan` → accumulate `log_xi_sum` and
     `log_gamma0_sum` with `jnp.logaddexp`
   - Apply M-step once per epoch

2. Add `--batch_size` flag to `ginc_experiment.py`.

3. Benchmark wall-clock time and final BPS vs full-batch on the DNA and GINC data.

**Expected outcome.** Mini-batch EM converges in similar iterations with lower
peak memory; enables scaling to large corpora.

---

## Goal 5: Probabilistic Next-Token Prediction (Full Posterior)

**Motivation.** `predict_next_obs` currently returns the argmax clone. For the
GINC ICL evaluation, outputting a full predictive distribution
P(o_{t+1} | o_{1:t}) would allow computing perplexity and proper log-likelihood
metrics, enabling comparison with neural LM baselines.

**Tasks:**

1. Add `predictive_distribution(obs_seq) -> jnp.ndarray (n_obs,)` to CSCG and
   SMCSCG:
   - Run forward pass to get `log_alpha[-1]`
   - For each obs o: sum `exp(log_alpha[-1, clones_o] + log_T[clones_o, :])` over
     all next-step clones, then marginalise to observation space

2. Update `ginc_experiment.py` to report:
   - Top-1 accuracy (existing)
   - Perplexity on held-out completions

3. Add a `--baseline uniform` flag that reports chance-level accuracy/perplexity.

---

## Ordering and Dependencies

| Goal | Depends on | Estimated scope |
|------|-----------|-----------------|
| 1 – SM-CSCG on GINC | Phase 1 done | Medium |
| 2 – Compression benchmarks | Phase 1 done | Small |
| 5 – Predictive distribution | Phase 1 done | Small |
| 3 – Dynamic K | Goal 5 useful for split criterion | Large |
| 4 – Mini-batch EM | Phase 1 done | Medium |

Recommended order: **2 → 5 → 1 → 4 → 3**
