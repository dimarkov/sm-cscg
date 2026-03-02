# Phase 2 Plan

## Context

Phase 1 delivered:
- JAX/Equinox CSCG and SM-CSCG with scan-vectorised Baum-Welch and Viterbi EM
- Coxian and general phase-type duration distributions
- GINC in-context learning experiment (replicating Swaminathan et al. NeurIPS 2023, CSCG only)
- Examples: symbol patterns, character text, DNA compression

Three experiments from the NeurIPS 2023 paper remain unfinished:
1. **Fast Rebinding Algorithm** (Algorithm 1) — the core ICL mechanism, not yet implemented
2. **LIALT** — Language Instructed Algorithm Learning Tasks
3. **Dax** — zero-shot novel word induction

Beyond the paper, there are engineering goals to improve compression quality and scalability.

---

## Goal 0: Fast Rebinding Algorithm (Algorithm 1)

**Motivation.** The paper's central contribution is not just training CSCG, but the
**fast rebinding** algorithm that adapts a grounded schema to a new context at test
time without full retraining. This is the mechanism that enables in-context learning.
It is required for both LIALT and Dax experiments.

**Algorithm sketch** (from paper Algorithm 1):

Given a grounded schema `{T, C, E^0}` and a prompt sequence `x_{1:t}`:

1. Run forward pass to get clone posteriors at each timestep.
2. **Anchors**: timesteps where the model predicts the correct token with high
   confidence (posterior > threshold). These tokens have known clone identity.
3. **Rebinding candidates**: timesteps where a non-ground token (novel/swapped
   symbol) is predicted with high confidence. These are the tokens needing rebinding.
4. For each rebinding candidate token `v`:
   - Collect the clone distribution at surrounding anchor timesteps.
   - Run a local EM update: update only the emission row(s) for `v` in `E`,
     using the anchor clone assignments as supervision.
5. Return the updated emission matrix `E'` (transition `T` stays fixed).

**Tasks:**

1. Add `rebind(schema, prompt, p_anchor=0.9, p_surprise=0.1, n_iter=10)`
   method to `CSCG`:
   - `schema` is a trained CSCG (the grounded model)
   - `prompt` is a list of observed tokens (context sequence)
   - Returns a new CSCG with updated emission matrix `E'`

2. Add a `predict_next_obs_rebind(schema, context, query)` convenience function
   that runs rebinding on `context` then predicts the next token for `query`.

3. Unit test: train CSCG on two interleaved schemas A/B; give context from schema B;
   verify rebinding recovers B's emissions.

**Expected outcome.** After rebinding on `n` context examples, the model correctly
assigns novel tokens to the right clones, enabling ICL without retraining.

---

## Goal 1: LIALT Experiment

**Motivation.** LIALT (Language Instructed Algorithm Learning Tasks) tests whether a
trained CSCG can perform algorithmic tasks (list reversal, matrix transpose, etc.) in
context. The model trains once on a corpus of all 13 algorithms, then at test time is
given either an instruction string or a few input/output examples and must execute the
correct algorithm.

**Dataset details** (from paper §4.2):
- 13 list/matrix algorithms (e.g., reverse, sort, rotate, transpose)
- Vocabulary: 676 tokens (all pairs of letters `aa`–`zz`)
- Training: generate sequences of algorithm executions; concatenate all algorithms
- `n_clones` proportional to token frequency ("overallocation ratio" hyperparameter)
- Training: 500 Baum-Welch iterations + 10 Viterbi iterations, ε=1e-6, p_surprise=0.1

**Test sets** (100 prompts each):
- **Instruction-based**: prompt = natural language instruction + query; predict output
- **Example-based**: prompt = `n` input/output pairs + query; predict output

**Tasks:**

1. Add `experiments/lialt_data.py`:
   - Implement the 13 algorithms
   - `generate_lialt_corpus(n_train, seed)` → list of token sequences
   - `generate_lialt_test(n_prompts, mode="instruction"|"example", n_context)` → prompts

2. Add `experiments/lialt_experiment.py`:
   - Train CSCG with overallocation ratio flag `--overalloc_ratio r`
     (n_clones[token] = round(r * freq[token]))
   - Apply fast rebinding (Goal 0) at test time
   - Report accuracy on both test sets as a function of n_context

3. Compare CSCG vs SM-CSCG on LIALT (SM-CSCG may handle variable-length
   algorithm outputs better).

**Expected outcome.** CSCG reaches the paper's reported accuracy on LIALT
instruction-based and example-based retrieval tasks.

---

## Goal 2: Dax Experiment

**Motivation.** The Dax experiment tests zero-shot novel word induction. A CSCG
trained on natural text can absorb the meaning of a new word (e.g., "dax") from a
single example sentence via fast rebinding, then correctly use it in fill-in-the-blank
probes.

**Setup** (from paper §4.3):
- Train corpus: PreCo dataset (English co-reference resolution corpus)
- Replace 5 frequent words with novel tokens (e.g., `<dax>`, `<blicket>`, ...)
- At test time: give 1–few example sentences containing the novel token in context
- Probe: fill-in-the-blank questions with the novel token as the answer

**Tasks:**

1. Add `experiments/dax_data.py`:
   - Download/load PreCo dataset
   - `make_dax_corpus(n_novel_words=5, seed)` → training corpus with novel tokens
   - `make_dax_test(n_prompts, n_context)` → (context, probe) pairs

2. Add `experiments/dax_experiment.py`:
   - Train CSCG on PreCo corpus (tokenized at word level)
   - At test: run rebinding (Goal 0) on context sentence(s)
   - Evaluate fill-in-the-blank accuracy

3. Report accuracy as a function of n_context (1, 2, 4, 8 examples).

**Expected outcome.** With even 1 example, rebinding correctly assigns the novel
token to its clone, enabling accurate fill-in-the-blank.

---

## Goal 3: SM-CSCG on Variable-Length GINC

**Motivation.** The GINC data generator produces sentences of fixed length `k`.
Phase-type durations add no value here. The interesting experiment is a
**variable-length GINC variant** where sentence lengths are drawn from a
non-geometric distribution (e.g., Poisson or Erlang). SM-CSCG should learn the
duration distribution and rebind schemas with fewer context examples.

**Tasks:**

1. Add a `--duration_dist` flag to `experiments/ginc_data.py`:
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

## Goal 4: Real Sequence Compression Benchmarks

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

## Goal 5: Probabilistic Next-Token Prediction (Full Posterior)

**Motivation.** `predict_next_obs` currently returns the argmax clone. Outputting a
full predictive distribution P(o_{t+1} | o_{1:t}) allows computing perplexity and
proper log-likelihood metrics, enabling comparison with neural LM baselines.

**Tasks:**

1. Add `predictive_distribution(obs_seq) -> jnp.ndarray (n_obs,)` to CSCG and
   SMCSCG:
   - Run forward pass to get `log_alpha[-1]`
   - For each obs o: sum `exp(log_alpha[-1, clones_o] + log_T[clones_o, :])` over
     all next-step clones, then marginalise to observation space

2. **Add next-token prediction accuracy to all examples** (`example_symbols.py`,
   `example_chartext.py`, `example_dna.py`):
   - For each sequence, compute `predict_next_obs` at every position t and compare
     to the actual next token o_{t+1}
   - Report train accuracy and test accuracy alongside BPS in the summary table
   - This gives a complementary view to BPS: BPS measures compression quality,
     accuracy measures prediction quality

3. Update `ginc_experiment.py` to report:
   - Top-1 accuracy (existing)
   - Perplexity on held-out completions

4. Add a `--baseline uniform` flag that reports chance-level accuracy/perplexity.

---

## Goal 6: Mini-batch / Online EM

**Motivation.** All current training loads all sequences into a single padded
batch. For long corpora (full chromosome, GINC with 10k documents) this exceeds
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

## Goal 7: Dynamic Clone Count (Variable K per Observation)

**Motivation.** The current implementation uses a fixed `n_clones` K for all
observations. In practice, frequent/ambiguous observations need more clones;
rare/unambiguous ones need fewer. The overallocation ratio used in LIALT is
a step in this direction.

**Tasks:**

1. Allow `n_clones` to be a list/array of length `n_obs` (one K per observation).
   Update `CSCG` / `SMCSCG` constructors and `_e_step_scan`.

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

## Ordering and Dependencies

| Goal | Depends on | Estimated scope |
|------|-----------|-----------------|
| 0 – Fast Rebinding Algorithm | Phase 1 done | Medium |
| 1 – LIALT | Goal 0 | Medium |
| 2 – Dax | Goal 0 | Medium |
| 3 – SM-CSCG on variable GINC | Phase 1 done | Medium |
| 4 – Compression benchmarks | Phase 1 done | Small |
| 5 – Predictive distribution | Phase 1 done | Small |
| 6 – Mini-batch EM | Phase 1 done | Medium |
| 7 – Dynamic K | Goal 5 useful for split criterion | Large |

Recommended order: **0 → 4 → 5 → 1 → 2 → 3 → 6 → 7**

Goals 0, 1, 2 are the paper-replication priorities.
Goals 4, 5, 6, 7 are engineering/benchmarking improvements.
