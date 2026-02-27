# Generative Models and EM Algorithms for CSCG and SM-CSCG

## Notation

| Symbol | Meaning |
|--------|---------|
| $N_o$ | Number of unique observations |
| $K$ | Number of clones per observation |
| $N = N_o \cdot K$ | Total number of macro hidden states (clone states) |
| $L$ | Number of phases per clone state (SM-CSCG) |
| $\tilde{N} = N \cdot L$ | Total expanded hidden states (SM-CSCG) |
| $o_t$ | Observation at time $t$ |
| $s_t$ | Hidden state at time $t$ |
| $\text{obs}(j)$ | The observation emitted by clone state $j$ |
| $C(o) = \{j : \text{obs}(j) = o\}$ | Set of clone states for observation $o$, $\|C(o)\| = K$ |
| $\tilde{C}(o) = \{(j,l) : \text{obs}(j) = o\}$ | Expanded clone set, $\|\tilde{C}(o)\| = K \cdot L$ |
| $T$ | Sequence length |

---

## 1. CSCG: Clone-Structured Cognitive Graph

### 1.1 Generative Model

CSCG is an overcomplete Hidden Markov Model with **deterministic, fixed emissions**.
Each observation $o$ has $K$ hidden "clone" states that all emit $o$ with probability 1, but
differ in their transition behaviour, enabling the model to disambiguate the same observation
appearing in different sequential contexts.

**Parameters:**

- Initial distribution: $\boldsymbol{\pi} \in \mathbb{R}^N$, $\;\pi_j = P(s_1 = j)$
- Transition matrix: $\mathbf{A} \in \mathbb{R}^{N \times N}$, $\;A_{ij} = P(s_t = j \mid s_{t-1} = i)$
- Emission matrix (**fixed**, never updated):

$$
E_{jo} = P(o_t = o \mid s_t = j) = \mathbf{1}[\text{obs}(j) = o]
$$

**Generative process:**

1. Draw initial state: $s_1 \sim \text{Categorical}(\boldsymbol{\pi})$
2. Emit: $o_1 = \text{obs}(s_1)$
3. For $t = 2, \ldots, T$:
   - Transition: $s_t \sim \text{Categorical}(\mathbf{A}_{s_{t-1}, \cdot})$
   - Emit: $o_t = \text{obs}(s_t)$

**Joint probability:**

$$
P(\mathbf{o}_{1:T}, \mathbf{s}_{1:T}) = \pi_{s_1} \prod_{t=2}^{T} A_{s_{t-1}, s_t} \prod_{t=1}^{T} E_{s_t, o_t}
$$

Since emissions are deterministic, the product $\prod_t E_{s_t, o_t}$ is either 1 (if every $s_t \in C(o_t)$) or 0.

### 1.2 Emission Sparsity

At time $t$, given observation $o_t$, only the $K$ clones in $C(o_t)$ have nonzero
forward probability. This reduces the effective per-step computation from
$O(N^2)$ to $O(N \cdot K)$, or $O(K^2)$ when both the source and target states
are restricted to their respective clone sets.

### 1.3 Forward-Backward Algorithm

All computations are performed in log-space to prevent underflow. Below we write
the probability-space equations; the implementation replaces products with sums
and sums with `logsumexp`.

**Forward variable:** $\alpha_t(j) = P(o_{1:t},\; s_t = j)$

*Initialization:*
$$
\alpha_1(j) = \begin{cases}
\pi_j & \text{if } j \in C(o_1) \\
0 & \text{otherwise}
\end{cases}
$$

*Recursion ($t = 2, \ldots, T$):*
$$
\alpha_t(j) = \begin{cases}
\displaystyle\sum_{i \in C(o_{t-1})} \alpha_{t-1}(i) \, A_{ij} & \text{if } j \in C(o_t) \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

The inner sum runs only over $i \in C(o_{t-1})$ because $\alpha_{t-1}(i) = 0$
for $i \notin C(o_{t-1})$.

*Likelihood:*
$$
P(\mathbf{o}_{1:T}) = \sum_{j \in C(o_T)} \alpha_T(j)
$$

**Backward variable:** $\beta_t(j) = P(o_{t+1:T} \mid s_t = j)$

*Terminal:* $\beta_T(j) = 1$

*Recursion ($t = T-1, \ldots, 1$):*
$$
\beta_t(j) = \sum_{i \in C(o_{t+1})} A_{ji} \, \beta_{t+1}(i)
$$

Only $j \in C(o_t)$ are needed for downstream computation.

### 1.4 EM Algorithm (Baum-Welch)

**E-step:** Compute forward-backward variables and the following posteriors.

*State posterior:*
$$
\gamma_t(j) = P(s_t = j \mid \mathbf{o}) = \frac{\alpha_t(j)\,\beta_t(j)}{P(\mathbf{o})}
$$

*Transition posterior:*
$$
\xi_t(i, j) = P(s_t = i,\, s_{t+1} = j \mid \mathbf{o}) = \frac{\alpha_t(i)\, A_{ij}\, E_{j,o_{t+1}}\, \beta_{t+1}(j)}{P(\mathbf{o})}
$$

Since $E_{j,o_{t+1}} = \mathbf{1}[j \in C(o_{t+1})]$, only pairs $(i, j) \in C(o_t) \times C(o_{t+1})$ are nonzero.

**M-step** (over all training sequences; subscript $s$ indexes sequences):

*Transitions:*
$$
\hat{A}_{ij} = \frac{\sum_s \sum_{t=1}^{T_s - 1} \xi_t^{(s)}(i, j)}{\sum_s \sum_{t=1}^{T_s - 1} \gamma_t^{(s)}(i)}
$$

*Initial distribution:*
$$
\hat{\pi}_j = \frac{\sum_s \gamma_1^{(s)}(j)}{\sum_s 1} = \frac{1}{S}\sum_s \gamma_1^{(s)}(j)
$$

*Emissions:* **Not updated.** The emission matrix $E$ is fixed by the clone structure.

---

## 2. SM-CSCG: Semi-Markov CSCG via Phase-Type Implicit Duration

### 2.1 Motivation: From Explicit to Implicit Duration

A standard HSMM models state duration with an **explicit** per-state distribution
$p_j(d)$, which requires a modified forward-backward algorithm with an inner
loop over durations, adding an $O(d_{\max})$ factor to inference cost.

The **implicit duration** approach instead represents each duration distribution
as a **discrete phase-type distribution** by expanding each macro-state into
multiple internal phases. The sojourn time in a macro-state equals the number
of time steps spent traversing its phases before exiting. This converts the HSMM
back into a standard HMM on an expanded state space, allowing the use of
unmodified Baum-Welch EM.

### 2.2 Discrete Phase-Type Distributions

A discrete phase-type (PH) distribution models the time to absorption in a
finite-state discrete-time Markov chain with one absorbing state. It is
parameterized by:

- **Entry distribution** $\boldsymbol{\alpha} \in \mathbb{R}^L$: probability of
  starting in each of $L$ transient phases, $\sum_l \alpha_l = 1$
- **Sub-transition matrix** $\mathbf{S} \in \mathbb{R}^{L \times L}$: transition
  probabilities among transient phases (sub-stochastic: row sums $\leq 1$)
- **Exit probability vector** $\mathbf{e} = \mathbf{1} - \mathbf{S}\mathbf{1}$:
  probability of absorption (exiting) from each phase

The PMF of the sojourn time $d$ (number of steps before absorption) is:

$$
p(d) = \boldsymbol{\alpha}^\top \mathbf{S}^{d-1} \mathbf{e}, \qquad d = 1, 2, 3, \ldots
$$

Phase-type distributions are dense in the set of all distributions on
$\{1, 2, \ldots\}$: any discrete distribution can be approximated arbitrarily
well by choosing $L$ large enough.

**Special cases:**

| Structure | Phases | Parameters | Implied duration |
|-----------|--------|------------|------------------|
| Single phase ($L=1$) | $\bullet \to \text{exit}$ | $e_1 = 1$ | Always $d=1$ (degenerate) |
| Geometric | $\bullet \circlearrowleft$ | $S_{11} = 1-q,\; e_1 = q$ | Geometric$(q)$, mean $1/q$ |
| Negative binomial | $\bullet \to \bullet \to \cdots \to \bullet$ | $S_{l,l+1} = 1-q$ | NegBin$(L, q)$, mean $L/q$ |
| Coxian | $\bullet \to \bullet \to \cdots \to \bullet$ with exit at each | $c_l, e_l$ per phase | Very flexible |
| General PH | Full $\mathbf{S}$ | $L^2 + L - 1$ params | Maximally flexible |

### 2.3 Generative Model

SM-CSCG extends CSCG by expanding each of the $N$ clone states into $L$
internal phases, yielding an HMM on $\tilde{N} = N \cdot L$ expanded states.
Each expanded state $(j, l)$ -- clone $j$, phase $l$ -- emits the same
observation $\text{obs}(j)$ deterministically. Duration in macro-state $j$ is
governed implicitly by a per-state phase-type distribution
$(\boldsymbol{\alpha}_j, \mathbf{S}_j)$.

**Parameters:**

- Macro-state initial distribution:
  $\boldsymbol{\pi} \in \mathbb{R}^N$, $\;\pi_j = P(\text{first macro-state} = j)$
- Macro-state transition matrix:
  $\mathbf{A} \in \mathbb{R}^{N \times N}$ with **no macro self-transitions**: $A_{jj} = 0$
- Per-state phase-type parameters (for each clone $j$):
  - Entry distribution: $\boldsymbol{\alpha}_j \in \mathbb{R}^L$, $\;\sum_l \alpha_{j,l} = 1$
  - Sub-transition matrix: $\mathbf{S}_j \in \mathbb{R}^{L \times L}$ (sub-stochastic)
  - Exit probabilities: $e_{j,l} = 1 - \sum_{l'} S_{j,ll'}$ (derived)
- Emission matrix (**fixed**):
  $\tilde{E}_{(j,l),\,o} = \mathbf{1}[\text{obs}(j) = o]$

These assemble into an **expanded HMM** with:

**Expanded initial distribution** $\tilde{\boldsymbol{\pi}} \in \mathbb{R}^{\tilde{N}}$:

$$
\tilde{\pi}_{(j,l)} = \pi_j \cdot \alpha_{j,l}
$$

**Expanded transition matrix** $\tilde{\mathbf{A}} \in \mathbb{R}^{\tilde{N} \times \tilde{N}}$:

$$
\tilde{A}_{(j,l),\,(j',l')} = \begin{cases}
S_{j,ll'} & \text{if } j' = j \quad \text{(intra-state phase transition)} \\[4pt]
e_{j,l} \cdot A_{jj'} \cdot \alpha_{j',l'} & \text{if } j' \neq j \quad \text{(inter-state transition)}
\end{cases}
$$

**Row-sum verification** (each row sums to 1):

$$
\sum_{(j',l')} \tilde{A}_{(j,l),(j',l')} =
\underbrace{\sum_{l'} S_{j,ll'}}_{\text{stay in }j}
+ \underbrace{e_{j,l}}_{\text{exit }j} \cdot
\underbrace{\sum_{j' \neq j} A_{jj'}}_{ = 1} \cdot
\underbrace{\sum_{l'} \alpha_{j',l'}}_{= 1}
= (1 - e_{j,l}) + e_{j,l} = 1
$$

**Generative process:**

1. Draw initial macro-state: $j_1 \sim \text{Categorical}(\boldsymbol{\pi})$
2. Draw initial phase: $l_1 \sim \text{Categorical}(\boldsymbol{\alpha}_{j_1})$
3. Set expanded state $\tilde{s}_1 = (j_1, l_1)$
4. Emit: $o_1 = \text{obs}(j_1)$
5. For $t = 2, \ldots, T$:
   - Transition: $\tilde{s}_t = (j_t, l_t) \sim \text{Categorical}(\tilde{\mathbf{A}}_{\tilde{s}_{t-1}, \cdot})$
     - With probability $S_{j_{t-1}, l_{t-1}, l_t}$: stay in same macro-state ($j_t = j_{t-1}$), advance to phase $l_t$
     - With probability $e_{j_{t-1}, l_{t-1}} \cdot A_{j_{t-1}, j_t} \cdot \alpha_{j_t, l_t}$: exit to new macro-state $j_t \neq j_{t-1}$, enter at phase $l_t$
   - Emit: $o_t = \text{obs}(j_t)$

**Joint probability:**

$$
P(\mathbf{o}_{1:T}, \tilde{\mathbf{s}}_{1:T}) = \tilde{\pi}_{\tilde{s}_1}
\prod_{t=2}^{T} \tilde{A}_{\tilde{s}_{t-1}, \tilde{s}_t}
\prod_{t=1}^{T} \tilde{E}_{\tilde{s}_t, o_t}
$$

This is a standard HMM joint probability on the expanded state space.

**Implied duration distribution** for macro-state $j$:

$$
p_j(d) = \boldsymbol{\alpha}_j^\top \, \mathbf{S}_j^{d-1} \, \mathbf{e}_j, \qquad d = 1, 2, \ldots
$$

Unlike the explicit-duration HSMM, there is no $d_{\max}$ truncation --
the phase-type support is unbounded (though probability decays
exponentially in $d$ for any finite $L$).

### 2.4 Emission Sparsity in the Expanded Model

All $L$ phases of clone $j$ emit $\text{obs}(j)$ deterministically. At time $t$,
given $o_t$, only expanded states in $\tilde{C}(o_t) = \{(j, l) : j \in C(o_t),\; l = 1, \ldots, L\}$
have nonzero forward probability, giving $K \cdot L$ active states per step.

The per-step cost using the block structure of $\tilde{\mathbf{A}}$ can be
decomposed efficiently (see Section 2.6).

### 2.5 Forward-Backward Algorithm

Since the expanded model is a standard HMM, inference uses standard
forward-backward with no HSMM modifications. All computations are in log-space.

**Forward variable:** $\tilde{\alpha}_t((j,l)) = P(o_{1:t},\; \tilde{s}_t = (j,l))$

*Initialization:*
$$
\tilde{\alpha}_1((j,l)) = \begin{cases}
\pi_j \cdot \alpha_{j,l} & \text{if } j \in C(o_1) \\
0 & \text{otherwise}
\end{cases}
$$

*Recursion ($t = 2, \ldots, T$):*
$$
\tilde{\alpha}_t((j',l')) = \begin{cases}
\displaystyle\sum_{(j,l) \in \tilde{C}(o_{t-1})} \tilde{\alpha}_{t-1}((j,l)) \, \tilde{A}_{(j,l),(j',l')} & \text{if } (j', l') \in \tilde{C}(o_t) \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

*Likelihood:*
$$
P(\mathbf{o}_{1:T}) = \sum_{(j,l) \in \tilde{C}(o_T)} \tilde{\alpha}_T((j,l))
$$

**Backward variable:** $\tilde{\beta}_t((j,l)) = P(o_{t+1:T} \mid \tilde{s}_t = (j,l))$

*Terminal:* $\tilde{\beta}_T((j,l)) = 1$

*Recursion ($t = T-1, \ldots, 1$):*
$$
\tilde{\beta}_t((j,l)) = \sum_{(j',l') \in \tilde{C}(o_{t+1})} \tilde{A}_{(j,l),(j',l')} \, \tilde{\beta}_{t+1}((j',l'))
$$

### 2.6 Efficient Forward Step via Block Decomposition

The naive per-step cost is $O(K^2 L^2)$. By exploiting the block structure of
$\tilde{\mathbf{A}}$, this reduces to $O(K L^2 + K^2)$.

Decompose the forward recursion for target $(j', l')$ with $j' \in C(o_t)$:

$$
\tilde{\alpha}_t((j',l')) =
\underbrace{\sum_{l} \tilde{\alpha}_{t-1}((j',l)) \, S_{j',ll'}}_{\text{intra-state}}
\;+\;
\underbrace{\alpha_{j',l'} \sum_{j \neq j',\, j \in C(o_{t-1})} A_{jj'} \,
\underbrace{\sum_{l} \tilde{\alpha}_{t-1}((j,l)) \, e_{j,l}}_{\text{exit mass}_j}
}_{\text{inter-state}}
$$

**Computation:**

1. **Exit mass** for each active source macro-state $j \in C(o_{t-1})$:
   $\quad m_j = \sum_{l=1}^{L} \tilde{\alpha}_{t-1}((j,l)) \cdot e_{j,l}$
   $\qquad$ Cost: $O(KL)$

2. **Inter-state entry** for each target macro-state $j' \in C(o_t)$:
   $\quad \mu_{j'} = \sum_{j \in C(o_{t-1}),\, j \neq j'} A_{jj'} \cdot m_j$
   $\qquad$ Cost: $O(K^2)$

3. **Intra-state phase transition** for each $(j', l')$ with $j' \in C(o_t)$:
   $\quad \tilde{\alpha}_t((j',l')) = \sum_{l} \tilde{\alpha}_{t-1}((j',l)) \, S_{j',ll'} + \alpha_{j',l'} \cdot \mu_{j'}$
   $\qquad$ Cost: $O(KL^2)$ for full $\mathbf{S}_j$; $O(KL)$ for Coxian

**Total per-step cost: $O(KL^2 + K^2)$**

For the Coxian structure where $\mathbf{S}_j$ is bidiagonal: **$O(KL + K^2)$**

### 2.7 EM Algorithm (Baum-Welch on Expanded HMM)

Since the expanded model is a standard HMM, EM uses standard Baum-Welch.
The M-step enforces the block structure of $\tilde{\mathbf{A}}$ by decomposing
the sufficient statistics into macro-transition, phase-transition, and
phase-entry components.

**E-step:** Compute $\tilde{\alpha}, \tilde{\beta}$ as in Section 2.5.

**Posteriors** (standard HMM):

*State posterior:*
$$
\tilde{\gamma}_t((j,l)) = \frac{\tilde{\alpha}_t((j,l)) \, \tilde{\beta}_t((j,l))}{P(\mathbf{o})}
$$

*Transition posterior:*
$$
\tilde{\xi}_t((j,l),(j',l')) = \frac{\tilde{\alpha}_t((j,l)) \, \tilde{A}_{(j,l),(j',l')} \, \tilde{E}_{(j',l'),o_{t+1}} \, \tilde{\beta}_{t+1}((j',l'))}{P(\mathbf{o})}
$$

**Sufficient statistics** (accumulated over all sequences, indexed by $s$):

#### Intra-state phase transitions

Expected number of phase transitions within macro-state $j$:

$$
n_{j,ll'}^{\text{intra}} = \sum_s \sum_{t=1}^{T_s - 1} \tilde{\xi}_t^{(s)}((j,l),(j,l'))
$$

#### Exit counts

Expected number of exits from phase $l$ of macro-state $j$:

$$
n_{j,l}^{\text{exit}} = \sum_s \sum_{t=1}^{T_s - 1} \sum_{j' \neq j} \sum_{l'} \tilde{\xi}_t^{(s)}((j,l),(j',l'))
$$

#### Macro-transition counts

Expected number of transitions from macro-state $j$ to macro-state $j'$:

$$
n_{jj'}^{\text{macro}} = \sum_s \sum_{t=1}^{T_s - 1} \sum_{l} \sum_{l'} \tilde{\xi}_t^{(s)}((j,l),(j',l'))
$$

#### Entry phase counts

Expected number of entries into macro-state $j'$ at phase $l'$:

$$
n_{j',l'}^{\text{entry}} = \sum_s \left[ \tilde{\gamma}_1^{(s)}((j',l')) + \sum_{t=1}^{T_s - 1} \sum_{j \neq j'} \sum_{l} \tilde{\xi}_t^{(s)}((j,l),(j',l')) \right]
$$

The first term captures initial entries; the second captures mid-sequence entries.

#### M-step

**Phase sub-transitions** (standard conditional MLE, restricted to same macro-state):

$$
\hat{S}_{j,ll'} = \frac{n_{j,ll'}^{\text{intra}}}{\sum_{l''} n_{j,ll''}^{\text{intra}} + n_{j,l}^{\text{exit}}}
$$

The exit probability is derived: $\hat{e}_{j,l} = 1 - \sum_{l'} \hat{S}_{j,ll'}$

**Macro-transitions** (aggregated over all phases, no self-transitions):

$$
\hat{A}_{jj'} = \begin{cases}
\displaystyle\frac{n_{jj'}^{\text{macro}}}{\sum_{j'' \neq j} n_{jj''}^{\text{macro}}} & \text{if } j' \neq j \\[8pt]
0 & \text{if } j' = j
\end{cases}
$$

**Entry distributions** (which phase is entered upon arrival at macro-state $j'$):

$$
\hat{\alpha}_{j',l'} = \frac{n_{j',l'}^{\text{entry}}}{\sum_{l''} n_{j',l''}^{\text{entry}}}
$$

**Macro initial distribution:**

$$
\hat{\pi}_j = \frac{\sum_l n_{j,l}^{\text{init}}}{\sum_{j'} \sum_l n_{j',l}^{\text{init}}}
\qquad \text{where } n_{j,l}^{\text{init}} = \sum_s \tilde{\gamma}_1^{(s)}((j,l))
$$

**Emissions:** Not updated. The emission matrix remains fixed by clone structure.

### 2.8 Coxian Phase-Type Specialization

The **Coxian** (left-to-right) structure is a practical default that balances
flexibility with parsimony. Phases form a chain with exit possible at each stage:

$$
\bullet_1 \xrightarrow{c_{j,1}} \bullet_2 \xrightarrow{c_{j,2}} \cdots \xrightarrow{c_{j,L-1}} \bullet_L \to \text{exit}
$$

**Constraints:**

- Always enter at phase 1: $\alpha_{j,1} = 1$, $\;\alpha_{j,l} = 0$ for $l > 1$
- Sub-transition matrix is bidiagonal:
  $S_{j,ll'} = c_{j,l} \cdot \mathbf{1}[l' = l+1]$ for $l < L$; $\;S_{j,Ll'} = 0$
- Exit probability: $e_{j,l} = 1 - c_{j,l}$ for $l < L$; $\;e_{j,L} = 1$
- Parameters per state: $L - 1$ continuation probabilities $c_{j,1}, \ldots, c_{j,L-1}$

**Implied duration PMF:**

$$
p_j(d) = \begin{cases}
(1 - c_{j,d}) \displaystyle\prod_{l=1}^{d-1} c_{j,l} & \text{if } d < L \\[6pt]
\displaystyle\prod_{l=1}^{L-1} c_{j,l} & \text{if } d = L
\end{cases}
$$

For $d > L$: $p_j(d) = 0$ (maximum sojourn = $L$ steps).

**Coxian M-step simplification:**

Since $\alpha_j$ is fixed and $\mathbf{S}_j$ is bidiagonal, the M-step reduces to:

$$
\hat{c}_{j,l} = \frac{n_{j,l,l+1}^{\text{intra}}}{n_{j,l,l+1}^{\text{intra}} + n_{j,l}^{\text{exit}}}
\qquad \text{for } l = 1, \ldots, L-1
$$

The forward step cost with Coxian structure is $O(KL + K^2)$ per time step.

### 2.9 Decoding (Viterbi)

Since the expanded model is a standard HMM, Viterbi decoding is standard.
The output is a sequence of expanded states $\tilde{s}_1, \ldots, \tilde{s}_T$,
from which macro-state segmentation is recovered by grouping consecutive
time steps with the same macro-state $j$:

$$
\tilde{s}_{1:T} = (\underbrace{(j_1, \cdot), \ldots, (j_1, \cdot)}_{d_1 \text{ steps}},\;
\underbrace{(j_2, \cdot), \ldots, (j_2, \cdot)}_{d_2 \text{ steps}},\; \ldots)
\quad \Rightarrow \quad
\text{segments} = [(j_1, 0, d_1),\; (j_2, d_1, d_2),\; \ldots]
$$

---

## 3. Comparison

| Aspect | CSCG | SM-CSCG (phase-type) |
|--------|------|----------------------|
| Base model | HMM | HMM (expanded state space) |
| State space | $N = N_o K$ | $\tilde{N} = N_o K L$ |
| Duration model | Geometric (via self-transitions) | Phase-type (implicit, via $L$ phases per state) |
| Self-transitions | Allowed (learned) | Macro: forbidden ($A_{jj}=0$). Phase: within-state only ($\mathbf{S}_j$) |
| Emissions | Fixed, deterministic | Fixed, deterministic (all phases of clone $j$ emit $\text{obs}(j)$) |
| Learned params | $\mathbf{A}, \boldsymbol{\pi}$ | $\mathbf{A}, \boldsymbol{\pi}, \{\mathbf{S}_j, \boldsymbol{\alpha}_j\}$ |
| Forward step cost | $O(K^2)$ | $O(KL^2 + K^2)$ general; $O(KL + K^2)$ Coxian |
| Inference algorithm | Standard HMM Baum-Welch | Standard HMM Baum-Welch (on expanded states) |
| Duration flexibility | Geometric only | Any distribution (PH-approximable) |
| Duration truncation | N/A | None needed (support is $\{1, \ldots, L\}$ for Coxian, unbounded for general) |
| Segmentation output | No | Yes (group consecutive same-macro-state steps) |

### Advantages of implicit (phase-type) over explicit duration HSMM

1. **Standard inference**: No HSMM-specific forward-backward needed. The model
   is a standard HMM, so existing HMM libraries and optimizations apply directly.

2. **No $d_{\max}$ truncation**: Explicit duration HSMMs truncate at $d_{\max}$,
   introducing a hard cutoff. Phase-type distributions have unbounded support
   (for general $\mathbf{S}_j$), with probability decaying exponentially.

3. **Computational efficiency**: Explicit HSMM costs $O(K^2 \cdot d_{\max})$ per step.
   Phase-type costs $O(KL^2 + K^2)$ with general phases or $O(KL + K^2)$ with
   Coxian. When $L \ll d_{\max}$ (a few phases can represent durations much larger
   than $L$), this is a significant win.

4. **Flexible duration families**: By choosing phase-type structure (Coxian, Erlang,
   general), duration shapes ranging from geometric to multimodal can be captured
   without changing the inference algorithm.

### When SM-CSCG outperforms CSCG

1. **Duration-carrying data**: When sojourn times are non-geometric and
   context-dependent, SM-CSCG captures these patterns directly.

2. **Segmentation tasks**: SM-CSCG produces macro-state segmentation with
   learned duration structure.

3. **Compression efficiency**: On data with structured sojourn times, SM-CSCG
   achieves lower bits-per-symbol.

### When they perform similarly

On data where each state is visited for exactly one time step, both models
are equivalent: SM-CSCG learns $e_{j,1} \approx 1$ (immediate exit from phase 1),
reducing to a standard HMM with the CSCG transition structure.
