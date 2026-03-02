"""Semi-Markov Clone-Structured Cognitive Graph (SM-CSCG) — JAX/Equinox.

Reuses the sparse-block forward/backward/xi_sum/viterbi from cscg_jax.py
on the expanded state space N_tilde = n_macro × n_phases
where n_macro = n_obs × n_clones.

Expanded state index = macro_j × n_phases + phase_l.
Active expanded states for obs o at one timestep: clones [o*n_clones : (o+1)*n_clones],
each with all n_phases phases → block size C_eff = n_clones × n_phases.

Two phase-type structures:
  - Coxian: left-to-right phases, entry always at phase 0.
  - General: full sub-transition matrix and entry distribution per macro-state.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import equinox as eqx
import numpy as np

import time

from .cscg import (_forward, _backward, _xi_sum, _viterbi, _log_normalize,
                        _pad_sequences, _e_step_scan, _viterbi_e_step_scan,
                        _predict_next_obs)


NEG_INF = -jnp.inf


# ---------------------------------------------------------------------------
# Build expanded transition matrix (in numpy, then convert to jax)
# ---------------------------------------------------------------------------

def _build_coxian_matrix(log_s, log_c, log_e, log_A, n_macro, n_phases):
    """Build (N_tilde, N_tilde) expanded transition matrix for Coxian phase-type.

    Each phase has three options (or two for the last phase):
      - stay in current phase (self-loop):  s[j,l]
      - advance to next phase:              c[j,l]   (l < L-1 only)
      - exit macro state:                   e[j,l]

    Parameters
    ----------
    log_s : (n_macro, n_phases)   log stay probs
    log_c : (n_macro, n_phases-1) log advance probs
    log_e : (n_macro, n_phases)   log exit probs
    log_A : (n_macro, n_macro)    log macro-transition probs (no self-loops)
    n_macro, n_phases : int

    Returns
    -------
    log_tilde_A : (N_tilde, N_tilde) jnp array
    """
    N, L = n_macro, n_phases
    NL = N * L
    log_tilde_A = np.full((NL, NL), float("-inf"))

    for j in range(N):
        for l in range(L):
            jl = j * L + l

            # Self-loop (stay in same phase)
            log_tilde_A[jl, jl] = log_s[j, l]

            # Advance to next phase
            if l < L - 1:
                jl1 = j * L + l + 1
                log_tilde_A[jl, jl1] = log_c[j, l]

            # Exit: inter-macro transitions (enter at phase 0)
            for jp in range(N):
                if jp == j:
                    continue
                jpl0 = jp * L + 0
                log_tilde_A[jl, jpl0] = log_e[j, l] + log_A[j, jp]

    return jnp.array(log_tilde_A)


def _build_general_matrix(log_S, log_alpha, log_A, n_macro, n_phases):
    """Build (N_tilde, N_tilde) expanded transition matrix for general phase-type.

    Parameters
    ----------
    log_S     : (n_macro, n_phases, n_phases) sub-transition matrices
    log_alpha : (n_macro, n_phases) entry distributions
    log_A     : (n_macro, n_macro) macro-transition probs
    """
    N, L = n_macro, n_phases
    NL = N * L
    log_tilde_A = np.full((NL, NL), float("-inf"))

    for j in range(N):
        for l in range(L):
            jl = j * L + l
            # Intra-state phase transitions
            for lp in range(L):
                jlp = j * L + lp
                log_tilde_A[jl, jlp] = log_S[j, l, lp]

            # Exit probability: 1 - sum S[j,l,:]
            log_stay = float(logsumexp(jnp.array(log_S[j, l])))
            if log_stay > -1e-10:
                log_e = float("-inf")
            else:
                log_e = float(jnp.log1p(-jnp.exp(jnp.array(log_stay))))

            for jp in range(N):
                if jp == j:
                    continue
                for lp in range(L):
                    jplp = jp * L + lp
                    log_tilde_A[jl, jplp] = log_e + log_A[j, jp] + log_alpha[jp, lp]

    return jnp.array(log_tilde_A)


# ---------------------------------------------------------------------------
# SMCSCG Equinox Module
# ---------------------------------------------------------------------------

class SMCSCG(eqx.Module):
    """Semi-Markov CSCG with phase-type implicit durations (JAX/Equinox).

    Parameters
    ----------
    n_obs      : number of unique observations
    n_clones   : clones per observation
    n_phases   : phases per clone (L)
    phase_type : 'coxian' or 'general'
    pseudocount: Laplace smoothing
    key        : jax PRNGKey
    """

    # Expanded HMM params (for forward/backward inference)
    log_tilde_A  : jax.Array    # (N_tilde, N_tilde)
    log_tilde_pi : jax.Array    # (N_tilde,)

    # Macro-level params (for structured M-step)
    log_A  : jax.Array          # (n_macro, n_macro) — no self-loops

    # Phase-type params (Coxian or General)
    log_s    : jax.Array        # (n_macro, n_phases) — Coxian stay probs
    log_c    : jax.Array        # (n_macro, n_phases-1) — Coxian advance probs
    log_e    : jax.Array        # (n_macro, n_phases) — Coxian exit probs
    log_S    : jax.Array        # (n_macro, n_phases, n_phases) — General only
    log_alpha: jax.Array        # (n_macro, n_phases) — General entry dist

    # Macro initial distribution
    log_pi : jax.Array          # (n_macro,)

    # Static fields
    n_obs      : int   = eqx.field(static=True)
    n_clones   : int   = eqx.field(static=True)
    n_phases   : int   = eqx.field(static=True)
    phase_type : str   = eqx.field(static=True)
    pseudocount: float = eqx.field(static=True)

    def __init__(self, n_obs, n_clones, n_phases=5,
                 phase_type="coxian", pseudocount=1e-6, mean_duration=None,
                 key=None):
        self.n_obs = n_obs
        self.n_clones = n_clones
        self.n_phases = n_phases
        self.phase_type = phase_type
        self.pseudocount = pseudocount

        if key is None:
            key = jax.random.PRNGKey(0)

        N = n_obs * n_clones   # n_macro
        L = n_phases

        keys = jax.random.split(key, 5)

        # Macro initial distribution: uniform
        self.log_pi = jnp.full(N, -jnp.log(N))

        # Macro transition matrix (no self-loops)
        raw_A = jax.random.exponential(keys[0], shape=(N, N))
        raw_A = raw_A.at[jnp.arange(N), jnp.arange(N)].set(0.0)
        log_A = jnp.log(raw_A + 1e-300) - logsumexp(
            jnp.log(raw_A + 1e-300), axis=1, keepdims=True
        )
        log_A = log_A.at[jnp.arange(N), jnp.arange(N)].set(float("-inf"))
        self.log_A = log_A

        if phase_type == "coxian":
            if mean_duration is not None and mean_duration > L:
                # Initialize to target mean duration:
                # stay prob s = 1 - L/mean_duration, split remaining into advance/exit
                s_base = 1.0 - L / mean_duration
                remain = 1.0 - s_base  # = L / mean_duration

                # Non-last phases: 70% advance, 30% exit
                probs_inner = np.zeros((N, L, 3))
                probs_inner[:, :, 0] = s_base       # stay
                probs_inner[:, :-1, 1] = 0.7 * remain  # advance (non-last)
                probs_inner[:, :-1, 2] = 0.3 * remain  # exit (non-last)
                probs_inner[:, -1, 1] = 0.0          # no advance for last phase
                probs_inner[:, -1, 2] = remain        # all remaining to exit

                # Add small random noise for symmetry breaking, then renormalize
                noise = 0.02 * np.array(jax.random.uniform(keys[1], shape=(N, L, 3)))
                noise[:, -1, 1] = 0.0  # keep advance=0 for last phase
                probs = np.array(probs_inner) + noise
                probs = probs / probs.sum(axis=2, keepdims=True)
                probs = jnp.array(probs)
            else:
                if mean_duration is not None and mean_duration <= L:
                    import warnings
                    warnings.warn(
                        f"mean_duration={mean_duration} <= n_phases={L}; "
                        f"falling back to random init"
                    )
                # Original random init
                raw = jax.random.exponential(keys[1], shape=(N, L, 3))
                raw = raw.at[:, -1, 1].set(0.0)
                row_sums = raw.sum(axis=2, keepdims=True)
                probs = raw / row_sums  # (N, L, 3): [stay, advance, exit]

            self.log_s = jnp.log(probs[:, :, 0])                # (N, L)
            self.log_c = jnp.log(probs[:, :L - 1, 1])           # (N, L-1)
            self.log_e = jnp.log(probs[:, :, 2])                # (N, L)
            # Unused General params (dummy)
            self.log_S = jnp.full((N, L, L), float("-inf"))
            self.log_alpha = jnp.full((N, L), -jnp.log(L))

        elif phase_type == "general":
            # Entry distribution: random Dirichlet
            raw_alpha = jax.random.exponential(keys[2], shape=(N, L))
            self.log_alpha = jnp.log(raw_alpha) - logsumexp(
                jnp.log(raw_alpha), axis=1, keepdims=True
            )
            # Sub-transition matrix: sub-stochastic rows
            raw_S = jax.random.exponential(keys[3], shape=(N, L, L))
            if mean_duration is not None and mean_duration > L:
                # Derive stay_frac from target mean duration (+ small noise)
                base = 1.0 - L / mean_duration
                noise = 0.05 * jax.random.uniform(keys[4], shape=(N, L, 1))
                stay_frac = jnp.clip(base + noise, 0.1, 0.99)
            else:
                if mean_duration is not None and mean_duration <= L:
                    import warnings
                    warnings.warn(
                        f"mean_duration={mean_duration} <= n_phases={L}; "
                        f"falling back to random init"
                    )
                stay_frac = 0.3 + 0.5 * jax.random.uniform(keys[4], shape=(N, L, 1))
            row_norms = logsumexp(jnp.log(raw_S), axis=2, keepdims=True)
            self.log_S = jnp.log(raw_S) - row_norms + jnp.log(stay_frac)
            # Unused Coxian params (dummy)
            self.log_s = jnp.full((N, L), float("-inf"))
            self.log_c = jnp.full((N, max(L - 1, 1)), float("-inf"))
            self.log_e = jnp.full((N, L), 0.0)
        else:
            raise ValueError(f"Unknown phase_type: {phase_type!r}")

        # Build expanded params
        log_tilde_A, log_tilde_pi = self._build_expanded_params()
        self.log_tilde_A = log_tilde_A
        self.log_tilde_pi = log_tilde_pi

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_macro(self):
        return self.n_obs * self.n_clones

    @property
    def n_states(self):
        return self.n_macro * self.n_phases

    @property
    def _c_eff(self):
        """Effective clone count per obs in expanded space."""
        return self.n_clones * self.n_phases

    # ------------------------------------------------------------------
    # Build expanded transition matrix
    # ------------------------------------------------------------------

    def _build_expanded_params(self):
        """Assemble log_tilde_A and log_tilde_pi from structured params."""
        N = self.n_macro
        L = self.n_phases

        # Convert to numpy for the loop-heavy build
        log_A_np = np.array(self.log_A)

        if self.phase_type == "coxian":
            log_s_np = np.array(self.log_s)
            log_c_np = np.array(self.log_c)
            log_e_np = np.array(self.log_e)
            log_tilde_A = _build_coxian_matrix(log_s_np, log_c_np, log_e_np,
                                               log_A_np, N, L)
            # Enter at phase 0
            log_tilde_pi = jnp.full(N * L, float("-inf"))
            log_tilde_pi = log_tilde_pi.at[jnp.arange(N) * L].set(self.log_pi)
        else:
            log_S_np = np.array(self.log_S)
            log_alpha_np = np.array(self.log_alpha)
            log_tilde_A = _build_general_matrix(log_S_np, log_alpha_np, log_A_np, N, L)
            # Entry distribution over phases
            log_tilde_pi = jnp.full(N * L, float("-inf"))
            for j in range(N):
                for l in range(L):
                    log_tilde_pi = log_tilde_pi.at[j * L + l].set(
                        self.log_pi[j] + self.log_alpha[j, l]
                    )

        return log_tilde_A, log_tilde_pi

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def log_likelihood(self, obs_seq):
        """Log P(obs_seq | model)."""
        obs = jnp.asarray(obs_seq, dtype=jnp.int32)
        _, log_Z = _forward(self.log_tilde_A, self.log_tilde_pi, obs, self._c_eff)
        return float(log_Z)

    def bps(self, obs_seq):
        """Bits per symbol."""
        ll = self.log_likelihood(obs_seq)
        return -ll / (len(obs_seq) * np.log(2))

    def decode(self, obs_seq):
        """Viterbi MAP decoding on expanded state space.

        Returns
        -------
        macro_states : ndarray (T,) int — macro-state (clone) indices
        segments     : list of (obs, start, duration) tuples
        """
        obs = jnp.asarray(obs_seq, dtype=jnp.int32)
        C_eff = self._c_eff
        L = self.n_phases
        expanded_states = np.array(
            _viterbi(self.log_tilde_A, self.log_tilde_pi, obs, C_eff)
        )

        # Convert expanded states to macro-states
        macro_states = expanded_states // L

        # Build segments: group consecutive same-macro-state runs
        segments = []
        T = len(obs_seq)
        t = 0
        while t < T:
            macro = int(macro_states[t])
            start = t
            while t < T and macro_states[t] == macro:
                t += 1
            duration = t - start
            segments.append((int(obs_seq[start]), start, duration))

        return macro_states, segments

    def predict_next_obs(self, obs_seq):
        """Predict P(x_{t+1} | x_{1:t}) for all t, in parallel.

        Returns
        -------
        log_probs : jax.Array (T, n_obs)
        """
        return _predict_next_obs(self.log_tilde_A, self.log_tilde_pi, obs_seq,
                                 self._c_eff, self.n_obs)

    def duration_pmf(self, macro_state, max_d=30):
        """Compute implied duration PMF P(D=d) for a given macro-state.

        Parameters
        ----------
        macro_state : int — macro-state index (0..n_macro-1)
        max_d       : int — maximum duration to compute

        Returns
        -------
        pmf : ndarray (max_d,)
        """
        j = macro_state
        L = self.n_phases
        pmf = np.zeros(max_d)

        if self.phase_type == "coxian":
            # Start at phase 0, propagate through phases
            # log_state[l] = log prob of being in phase l
            log_state = np.full(L, float("-inf"))
            log_state[0] = 0.0
            for d in range(1, max_d + 1):
                # Prob of exiting at this step = sum_l P(phase l) * e[j,l]
                log_exit = float("-inf")
                for l in range(L):
                    log_exit = float(jnp.logaddexp(
                        jnp.array(log_exit),
                        jnp.array(log_state[l] + float(self.log_e[j, l]))
                    ))
                pmf[d - 1] = np.exp(log_exit)

                # Propagate: stay in same phase or advance to next
                new_log_state = np.full(L, float("-inf"))
                for l in range(L):
                    # Stay
                    if log_state[l] > float("-inf"):
                        new_log_state[l] = float(jnp.logaddexp(
                            jnp.array(new_log_state[l]),
                            jnp.array(log_state[l] + float(self.log_s[j, l]))
                        ))
                    # Advance
                    if l < L - 1 and log_state[l] > float("-inf"):
                        new_log_state[l + 1] = float(jnp.logaddexp(
                            jnp.array(new_log_state[l + 1]),
                            jnp.array(log_state[l] + float(self.log_c[j, l]))
                        ))
                log_state = new_log_state

        elif self.phase_type == "general":
            # Simulate via matrix powers
            S_j = np.exp(np.array(self.log_S[j]))  # (L, L)
            alpha_j = np.exp(np.array(self.log_alpha[j]))  # (L,)
            e_j = 1.0 - S_j.sum(axis=1)  # exit probs (L,)
            v = alpha_j.copy()
            for d in range(1, max_d + 1):
                pmf[d - 1] = float(v @ e_j)
                v = v @ S_j

        return pmf

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, sequences, n_iter=50, tol=1e-4, verbose=False,
            em_method="baum-welch"):
        """Baum-Welch EM or Viterbi EM with structured M-step.

        Parameters
        ----------
        em_method : str
            'baum-welch' (default) or 'viterbi'.

        Returns
        -------
        model : SMCSCG
        log_likelihoods : list[float]
        """
        if em_method == "viterbi":
            return self.fit_viterbi(sequences, n_iter=n_iter, tol=tol,
                                    verbose=verbose)
        model = self
        log_likelihoods = []

        for iteration in range(n_iter):
            total_ll, log_xi_sum, log_gamma0_sum = model._e_step(sequences, verbose=verbose)
            log_likelihoods.append(total_ll)
            if verbose:
                print(f"  SM-CSCG BW iter {iteration:3d}: LL = {total_ll:.4f}")
            if iteration > 0 and total_ll - log_likelihoods[-2] < tol:
                break
            model = model._m_step(log_xi_sum, log_gamma0_sum)

        return model, log_likelihoods

    def fit_viterbi(self, sequences, n_iter=50, tol=1e-4, verbose=False):
        """Viterbi EM (hard MAP assignments) with structured M-step.

        Returns
        -------
        model : SMCSCG
        log_likelihoods : list[float]
        """
        model = self
        log_likelihoods = []
        N_tilde = self.n_states
        C_eff = self._c_eff

        for iteration in range(n_iter):
            obs_batch, true_lens = _pad_sequences(sequences)
            total_ll, log_xi_sum, log_gamma0_sum = _viterbi_e_step_scan(
                model.log_tilde_A, model.log_tilde_pi,
                obs_batch, true_lens, C_eff, N_tilde
            )
            total_ll = float(total_ll)
            log_likelihoods.append(total_ll)
            if verbose:
                delta = (f"  Δ={total_ll - log_likelihoods[-2]:+.2f}"
                         if iteration > 0 else "")
                print(f"  SM-CSCG Vit iter {iteration:3d}: LL={total_ll:.2f}{delta}")
            if iteration > 0 and total_ll - log_likelihoods[-2] < tol:
                break
            model = model._m_step(log_xi_sum, log_gamma0_sum)

        return model, log_likelihoods

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self, sequences, verbose=False):
        """Forward-backward on all sequences; accumulate sufficient statistics."""
        N_tilde = self.n_states
        C_eff = self._c_eff
        if verbose:
            print(f"    [scan {len(sequences)} seqs] compiling/running...")
        t0 = time.time()
        obs_batch, true_lens = _pad_sequences(sequences)
        total_ll, log_xi_sum, log_gamma0_sum = _e_step_scan(
            self.log_tilde_A, self.log_tilde_pi, obs_batch, true_lens, C_eff, N_tilde
        )
        if verbose:
            print(f"    done {time.time()-t0:.1f}s  LL={float(total_ll):.2f}")
        return float(total_ll), log_xi_sum, log_gamma0_sum

    # ------------------------------------------------------------------
    # M-step (structured)
    # ------------------------------------------------------------------

    def _m_step(self, log_xi_sum, log_gamma0_sum):
        """Structured M-step: update phase-type params and macro-transitions."""
        if self.phase_type == "coxian":
            return self._m_step_coxian(log_xi_sum, log_gamma0_sum)
        else:
            return self._m_step_general(log_xi_sum, log_gamma0_sum)

    def _m_step_coxian(self, log_xi_sum, log_gamma0_sum):
        """M-step for Coxian phase-type with stay/advance/exit."""
        N = self.n_macro
        L = self.n_phases
        eps = max(self.pseudocount, 1e-300)
        log_eps = jnp.log(jnp.array(eps))
        xi = np.array(log_xi_sum)

        new_log_s = np.full((N, L), float("-inf"))
        new_log_c = np.full((N, max(L - 1, 1)), float("-inf"))
        new_log_e = np.full((N, L), float("-inf"))
        new_log_A = np.full((N, N), float("-inf"))

        for j in range(N):
            for l in range(L):
                jl = j * L + l

                # Stay (self-loop)
                log_stay = xi[jl, jl]

                # Advance to next phase
                if l < L - 1:
                    jl1 = j * L + l + 1
                    log_adv = xi[jl, jl1]
                else:
                    log_adv = float("-inf")

                # Exit: sum of xi to any other macro-state
                log_exit_terms = []
                for jp in range(N):
                    if jp == j:
                        continue
                    jpl0 = jp * L + 0
                    if xi[jl, jpl0] > float("-inf"):
                        log_exit_terms.append(xi[jl, jpl0])
                if log_exit_terms:
                    log_exit = float(logsumexp(jnp.array(log_exit_terms)))
                else:
                    log_exit = float(log_eps)

                # Normalize stay/advance/exit (or stay/exit for last phase)
                if l < L - 1:
                    terms = jnp.array([log_stay, log_adv, log_exit])
                    terms = jnp.logaddexp(terms, log_eps)
                    log_total = logsumexp(terms)
                    new_log_s[j, l] = float(terms[0] - log_total)
                    new_log_c[j, l] = float(terms[1] - log_total)
                    new_log_e[j, l] = float(terms[2] - log_total)
                else:
                    terms = jnp.array([log_stay, log_exit])
                    terms = jnp.logaddexp(terms, log_eps)
                    log_total = logsumexp(terms)
                    new_log_s[j, l] = float(terms[0] - log_total)
                    new_log_e[j, l] = float(terms[1] - log_total)

                # Accumulate macro-transitions
                for jp in range(N):
                    if jp == j:
                        continue
                    jpl0 = jp * L + 0
                    new_log_A[j, jp] = float(jnp.logaddexp(
                        jnp.array(new_log_A[j, jp]),
                        jnp.array(xi[jl, jpl0])
                    ))

        # Normalize log_A rows
        for j in range(N):
            row = jnp.array(new_log_A[j])
            row = jnp.logaddexp(row, log_eps)
            row = row.at[j].set(float("-inf"))
            row_sum = logsumexp(row)
            new_log_A[j] = np.array(row - row_sum)
            new_log_A[j, j] = float("-inf")

        # Update initial distribution from gamma0
        new_log_pi = np.full(N, float("-inf"))
        for j in range(N):
            g0_j = float("-inf")
            for l in range(L):
                g0_j = float(jnp.logaddexp(
                    jnp.array(g0_j),
                    jnp.array(float(log_gamma0_sum[j * L + l]))
                ))
            new_log_pi[j] = g0_j

        new_log_pi_arr = jnp.logaddexp(jnp.array(new_log_pi), log_eps)
        new_log_pi_arr = _log_normalize(new_log_pi_arr)

        # Build new model with updated params
        new_model = eqx.tree_at(
            lambda m: (m.log_s, m.log_c, m.log_e, m.log_A, m.log_pi),
            self,
            (jnp.array(new_log_s), jnp.array(new_log_c),
             jnp.array(new_log_e), jnp.array(new_log_A), new_log_pi_arr),
        )
        # Rebuild expanded matrices
        log_tilde_A, log_tilde_pi = new_model._build_expanded_params()
        new_model = eqx.tree_at(
            lambda m: (m.log_tilde_A, m.log_tilde_pi),
            new_model,
            (log_tilde_A, log_tilde_pi),
        )
        return new_model

    def _m_step_general(self, log_xi_sum, log_gamma0_sum):
        """M-step for general phase-type."""
        N = self.n_macro
        L = self.n_phases
        eps = max(self.pseudocount, 1e-300)
        log_eps = jnp.log(jnp.array(eps))
        xi = np.array(log_xi_sum)

        new_log_S = np.full((N, L, L), float("-inf"))
        new_log_alpha = np.full((N, L), float("-inf"))
        new_log_A = np.full((N, N), float("-inf"))
        new_log_pi = np.full(N, float("-inf"))

        for j in range(N):
            # Entry distribution: from gamma0 (what phase entered at t=0)
            for l in range(L):
                new_log_alpha[j, l] = float(log_gamma0_sum[j * L + l])

            # Sub-transition S[j, l, l'] = xi[(j,l), (j,l')] / row_sum
            for l in range(L):
                jl = j * L + l
                row = np.array([xi[jl, j * L + lp] for lp in range(L)])
                new_log_S[j, l] = row

            # Macro-transitions
            for jp in range(N):
                if jp == j:
                    continue
                log_p = float("-inf")
                for l in range(L):
                    jl = j * L + l
                    for lp in range(L):
                        jplp = jp * L + lp
                        log_p = float(jnp.logaddexp(
                            jnp.array(log_p), jnp.array(xi[jl, jplp])
                        ))
                new_log_A[j, jp] = log_p

            # Initial distribution
            g0_j = float("-inf")
            for l in range(L):
                g0_j = float(jnp.logaddexp(
                    jnp.array(g0_j),
                    jnp.array(float(log_gamma0_sum[j * L + l]))
                ))
            new_log_pi[j] = g0_j

        # Normalize S rows
        for j in range(N):
            for l in range(L):
                row = jnp.logaddexp(jnp.array(new_log_S[j, l]), log_eps)
                row_sum = logsumexp(row)
                new_log_S[j, l] = np.array(row - row_sum)

        # Normalize alpha
        for j in range(N):
            row = jnp.logaddexp(jnp.array(new_log_alpha[j]), log_eps)
            new_log_alpha[j] = np.array(row - logsumexp(row))

        # Normalize A rows (no self-loops)
        for j in range(N):
            row = jnp.array(new_log_A[j])
            row = jnp.logaddexp(row, log_eps)
            row = row.at[j].set(float("-inf"))
            new_log_A[j] = np.array(row - logsumexp(row))
            new_log_A[j, j] = float("-inf")

        # Normalize pi
        new_log_pi_arr = _log_normalize(jnp.logaddexp(jnp.array(new_log_pi), log_eps))

        new_model = eqx.tree_at(
            lambda m: (m.log_S, m.log_alpha, m.log_A, m.log_pi),
            self,
            (jnp.array(new_log_S), jnp.array(new_log_alpha),
             jnp.array(new_log_A), new_log_pi_arr),
        )
        log_tilde_A, log_tilde_pi = new_model._build_expanded_params()
        new_model = eqx.tree_at(
            lambda m: (m.log_tilde_A, m.log_tilde_pi),
            new_model,
            (log_tilde_A, log_tilde_pi),
        )
        return new_model
