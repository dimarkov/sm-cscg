"""Clone-Structured Cognitive Graph (CSCG) — JAX/Equinox implementation.

Sparse-block design: at each timestep only C = n_clones states are active
(the clones for the current observation). Forward/backward use jax.lax.scan
carrying a C-vector; xi accumulation updates C×C blocks in an (N,N) matrix
via jax.lax.dynamic_update_slice.

E-step: all sequences are padded to the same length and processed in a single
jax.lax.scan (one JIT-compiled XLA program, no Python loop overhead).

Reference: George et al. (2021); port of
  github.com/google-deepmind/space_is_a_latent_sequence (JAX, log-space).
"""

import functools
import time

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import equinox as eqx
import numpy as np


NEG_INF = -jnp.inf


# ---------------------------------------------------------------------------
# Module-level JIT helpers (pure functions; can be jitted/vmapped freely)
# ---------------------------------------------------------------------------

@jax.jit
def _log_normalize(v):
    """Normalize a log-prob vector so logsumexp(result) == 0."""
    return v - logsumexp(v)


def _forward(log_T, log_pi, obs_seq, n_clones):
    """Forward pass in log-space using jax.lax.scan (sparse-block).

    Parameters
    ----------
    log_T   : (N, N) float32
    log_pi  : (N,)   float32
    obs_seq : (T,)   int32
    n_clones: int    (static)

    Returns
    -------
    log_alphas_C : (T, n_clones) — alpha restricted to active clones
    log_Z        : scalar        — log P(obs_seq)
    """
    C = n_clones
    T = obs_seq.shape[0]

    o0 = obs_seq[0]
    log_alpha_C = jax.lax.dynamic_slice(log_pi, [o0 * C], [C])

    def step(carry, t):
        log_alpha_C, obs_prev = carry
        obs_cur = obs_seq[t]
        T_block = jax.lax.dynamic_slice(log_T, [obs_prev * C, obs_cur * C], [C, C])
        log_alpha_new = logsumexp(log_alpha_C[:, None] + T_block, axis=0)
        return (log_alpha_new, obs_cur), log_alpha_new

    ts = jnp.arange(1, T)
    (_, _), log_alphas_rest = jax.lax.scan(step, (log_alpha_C, o0), ts)
    log_alphas_C = jnp.concatenate([log_alpha_C[None, :], log_alphas_rest], axis=0)
    log_Z = logsumexp(log_alphas_C[-1])
    return log_alphas_C, log_Z


def _forward_masked(log_T, log_pi, obs_seq, n_clones, true_len):
    """Forward pass for a padded sequence; log_Z is read at true_len-1, not -1."""
    C = n_clones
    log_alphas_C, _ = _forward(log_T, log_pi, obs_seq, n_clones)
    # Read from position true_len-1 (dynamic, traced)
    log_alpha_last = jax.lax.dynamic_slice(log_alphas_C, [true_len - 1, 0], [1, C])
    log_Z = logsumexp(log_alpha_last[0])
    return log_alphas_C, log_Z


def _backward(log_T, obs_seq, n_clones):
    """Backward pass in log-space using jax.lax.scan (sparse-block).

    Returns
    -------
    log_betas_C : (T, n_clones) — beta restricted to active clones
    """
    C = n_clones
    T = obs_seq.shape[0]

    log_beta_C = jnp.zeros(C)

    def step(carry, t):
        log_beta_next, obs_next = carry
        obs_cur = obs_seq[t]
        T_block = jax.lax.dynamic_slice(log_T, [obs_cur * C, obs_next * C], [C, C])
        log_beta_cur = logsumexp(T_block + log_beta_next[None, :], axis=1)
        return (log_beta_cur, obs_cur), log_beta_cur

    ts = jnp.arange(T - 2, -1, -1)
    obs_last = obs_seq[-1]
    (_, _), log_betas_rest = jax.lax.scan(step, (log_beta_C, obs_last), ts)
    log_betas_rest = log_betas_rest[::-1]  # (T-1, C), t=0..T-2

    log_betas_C = jnp.concatenate([log_betas_rest, log_beta_C[None, :]], axis=0)
    return log_betas_C


def _xi_sum(log_alphas_C, obs_seq, log_T, log_betas_C, log_Z, n_clones, n_states):
    """Accumulate pairwise sufficient statistics into an (N, N) log matrix.

    Uses jax.lax.dynamic_update_slice to scatter C×C blocks.

    Returns
    -------
    log_xi_sum : (N, N)
    """
    C = n_clones
    N = n_states
    T = obs_seq.shape[0]

    log_xi_init = jnp.full((N, N), NEG_INF)

    def step(log_xi_sum, t):
        obs_t  = obs_seq[t]
        obs_t1 = obs_seq[t + 1]
        log_alpha = log_alphas_C[t]
        log_beta  = log_betas_C[t + 1]
        T_block   = jax.lax.dynamic_slice(log_T, [obs_t * C, obs_t1 * C], [C, C])
        log_xi_t  = log_alpha[:, None] + T_block + log_beta[None, :] - log_Z

        existing  = jax.lax.dynamic_slice(log_xi_sum, [obs_t * C, obs_t1 * C], [C, C])
        log_xi_sum = jax.lax.dynamic_update_slice(
            log_xi_sum, jnp.logaddexp(existing, log_xi_t), [obs_t * C, obs_t1 * C]
        )
        return log_xi_sum, None

    ts = jnp.arange(T - 1)
    log_xi_sum, _ = jax.lax.scan(step, log_xi_init, ts)
    return log_xi_sum


def _xi_sum_masked(log_alphas_C, obs_seq, log_T, log_betas_C, log_Z,
                   n_clones, n_states, true_len):
    """Like _xi_sum but skips timesteps >= true_len-1 (padding mask)."""
    C = n_clones
    N = n_states
    T = obs_seq.shape[0]

    log_xi_init = jnp.full((N, N), NEG_INF)

    def step(log_xi_sum, t):
        obs_t  = obs_seq[t]
        obs_t1 = obs_seq[t + 1]
        log_alpha = log_alphas_C[t]
        log_beta  = log_betas_C[t + 1]
        T_block   = jax.lax.dynamic_slice(log_T, [obs_t * C, obs_t1 * C], [C, C])
        log_xi_t  = log_alpha[:, None] + T_block + log_beta[None, :] - log_Z

        existing = jax.lax.dynamic_slice(log_xi_sum, [obs_t * C, obs_t1 * C], [C, C])
        # Only accumulate if within the true sequence (mask out padding)
        valid   = t < true_len - 1
        updated = jnp.where(valid, jnp.logaddexp(existing, log_xi_t), existing)
        log_xi_sum = jax.lax.dynamic_update_slice(
            log_xi_sum, updated, [obs_t * C, obs_t1 * C]
        )
        return log_xi_sum, None

    ts = jnp.arange(T - 1)
    log_xi_sum, _ = jax.lax.scan(step, log_xi_init, ts)
    return log_xi_sum


def _pad_sequences(sequences):
    """Pad sequences to the same length by repeating the last token.

    Returns
    -------
    obs_batch : jnp.array (B, max_len) int32
    true_lens : jnp.array (B,)         int32
    """
    max_len   = max(len(s) for s in sequences)
    true_lens = np.array([len(s) for s in sequences], dtype=np.int32)
    padded    = []
    for s in sequences:
        arr   = np.asarray(s, dtype=np.int32)
        pad_n = max_len - len(arr)
        padded.append(np.concatenate([arr, np.full(pad_n, arr[-1], dtype=np.int32)]))
    obs_batch = jnp.stack([jnp.asarray(p) for p in padded])   # (B, max_len)
    return obs_batch, jnp.asarray(true_lens)


@functools.partial(jax.jit, static_argnums=(4, 5))
def _e_step_scan(log_T, log_pi, obs_batch, true_lens, n_clones, n_states):
    """JIT-compiled E-step: scan over a batch of (padded) sequences.

    Parameters
    ----------
    log_T      : (N, N)
    log_pi     : (N,)
    obs_batch  : (B, max_len) int32
    true_lens  : (B,)         int32
    n_clones, n_states : static ints

    Returns
    -------
    total_ll       : scalar
    log_xi_sum     : (N, N)
    log_gamma0_sum : (N,)
    """
    C, N = n_clones, n_states

    def step(carry, inputs):
        log_xi_sum, log_gamma0_sum = carry
        obs_seq, true_len = inputs

        log_alphas_C, log_Z = _forward_masked(log_T, log_pi, obs_seq, C, true_len)
        log_betas_C         = _backward(log_T, obs_seq, C)

        log_xi_sum = jnp.logaddexp(
            log_xi_sum,
            _xi_sum_masked(log_alphas_C, obs_seq, log_T, log_betas_C,
                           log_Z, C, N, true_len),
        )

        o0           = obs_seq[0]
        log_gamma0_C = log_alphas_C[0] + log_betas_C[0] - log_Z
        existing     = jax.lax.dynamic_slice(log_gamma0_sum, [o0 * C], [C])
        log_gamma0_sum = jax.lax.dynamic_update_slice(
            log_gamma0_sum, jnp.logaddexp(existing, log_gamma0_C), [o0 * C]
        )
        return (log_xi_sum, log_gamma0_sum), log_Z

    init = (jnp.full((N, N), NEG_INF), jnp.full(N, NEG_INF))
    (log_xi_sum, log_gamma0_sum), log_Zs = jax.lax.scan(
        step, init, (obs_batch, true_lens)
    )
    return log_Zs.sum(), log_xi_sum, log_gamma0_sum


@functools.partial(jax.jit, static_argnums=(4, 5))
def _viterbi_e_step_scan(log_T, log_pi, obs_batch, true_lens, n_clones, n_states):
    """JIT-compiled Viterbi E-step: scan over padded sequences.

    Accumulates hard counts (log 1 = 0.0) from Viterbi paths into
    log_xi_sum / log_gamma0_sum, masking padded timesteps via true_lens.

    Returns
    -------
    total_ll       : scalar  (sum of forward log_Z over sequences)
    log_xi_sum     : (N, N)
    log_gamma0_sum : (N,)
    """
    C, N = n_clones, n_states

    def seq_step(carry, inputs):
        log_xi_sum, log_gamma0_sum = carry
        obs_seq, true_len = inputs

        states   = _viterbi(log_T, log_pi, obs_seq, C)
        _, log_Z = _forward(log_T, log_pi, obs_seq, C)

        # gamma0: hard count at initial state (t=0 is never padded)
        s0 = states[0]
        log_gamma0_sum = log_gamma0_sum.at[s0].set(
            jnp.logaddexp(log_gamma0_sum[s0], 0.0)
        )

        # xi: hard count for each consecutive pair, masked by true_len
        def xi_step(log_xi_sum, t):
            si    = states[t]
            sj    = states[t + 1]
            valid = t < true_len - 1
            updated = jnp.where(valid,
                                 jnp.logaddexp(log_xi_sum[si, sj], 0.0),
                                 log_xi_sum[si, sj])
            return log_xi_sum.at[si, sj].set(updated), None

        ts = jnp.arange(obs_seq.shape[0] - 1)
        log_xi_sum, _ = jax.lax.scan(xi_step, log_xi_sum, ts)
        return (log_xi_sum, log_gamma0_sum), log_Z

    init = (jnp.full((N, N), NEG_INF), jnp.full(N, NEG_INF))
    (log_xi_sum, log_gamma0_sum), log_Zs = jax.lax.scan(
        seq_step, init, (obs_batch, true_lens)
    )
    return log_Zs.sum(), log_xi_sum, log_gamma0_sum


def _viterbi(log_T, log_pi, obs_seq, n_clones):
    """Viterbi decoding (max-product) via two scans.

    Returns
    -------
    states : (T,) int32 — most likely hidden state sequence
    """
    C = n_clones
    T = obs_seq.shape[0]

    o0 = obs_seq[0]
    log_delta_C = jax.lax.dynamic_slice(log_pi, [o0 * C], [C])

    def fwd_step(carry, t):
        log_delta_prev, obs_prev = carry
        obs_cur = obs_seq[t]
        T_block = jax.lax.dynamic_slice(log_T, [obs_prev * C, obs_cur * C], [C, C])
        scores          = log_delta_prev[:, None] + T_block
        best_prev_local = jnp.argmax(scores, axis=0)
        log_delta_cur   = scores[best_prev_local, jnp.arange(C)]
        best_prev_global = obs_prev * C + best_prev_local
        return (log_delta_cur, obs_cur), (log_delta_cur, best_prev_global, obs_cur)

    ts = jnp.arange(1, T)
    (log_delta_last, obs_last), (log_deltas, best_prevs, obs_ats) = jax.lax.scan(
        fwd_step, (log_delta_C, o0), ts
    )

    best_local_last = jnp.argmax(log_delta_last)
    best_state_last = obs_last * C + best_local_last

    def bwd_step(state_next, t):
        obs_cur    = obs_ats[t]
        bp         = best_prevs[t]
        local_next = state_next - obs_cur * C
        state_t    = bp[local_next]
        return state_t, state_t

    bt_indices = jnp.arange(T - 2, -1, -1)
    _, states_rev = jax.lax.scan(bwd_step, best_state_last, bt_indices)

    states = jnp.concatenate([states_rev[::-1], jnp.array([best_state_last])])
    return states


# ---------------------------------------------------------------------------
# CSCG Equinox Module
# ---------------------------------------------------------------------------

class CSCG(eqx.Module):
    """Clone-Structured Cognitive Graph (JAX/Equinox).

    Parameters
    ----------
    n_obs       : number of unique observations
    n_clones    : clones per observation (uniform)
    pseudocount : Laplace smoothing added to sufficient statistics
    key         : jax.random.PRNGKey for initialization
    """

    log_T: jax.Array        # (n_states, n_states)
    log_pi: jax.Array       # (n_states,)
    n_obs: int              = eqx.field(static=True)
    n_clones: int           = eqx.field(static=True)
    pseudocount: float      = eqx.field(static=True)

    def __init__(self, n_obs, n_clones, pseudocount=1e-6, key=None):
        self.n_obs = n_obs
        self.n_clones = n_clones
        self.pseudocount = pseudocount

        if key is None:
            key = jax.random.PRNGKey(0)

        N  = n_obs * n_clones
        k1, _ = jax.random.split(key)
        raw    = jax.random.uniform(k1, shape=(N, N))
        log_T  = jnp.log(raw) - logsumexp(jnp.log(raw), axis=1, keepdims=True)
        self.log_T  = log_T
        self.log_pi = jnp.full(N, -jnp.log(N))

    @property
    def n_states(self):
        return self.n_obs * self.n_clones

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def log_likelihood(self, obs_seq):
        """Log P(obs_seq | model)."""
        obs = jnp.asarray(obs_seq, dtype=jnp.int32)
        _, log_Z = _forward(self.log_T, self.log_pi, obs, self.n_clones)
        return float(log_Z)

    def bps(self, obs_seq):
        """Bits per symbol = -log_likelihood / (T * ln 2)."""
        return -self.log_likelihood(obs_seq) / (len(obs_seq) * np.log(2))

    def decode(self, obs_seq):
        """Viterbi MAP decoding → ndarray (T,) int32."""
        obs = jnp.asarray(obs_seq, dtype=jnp.int32)
        return np.array(_viterbi(self.log_T, self.log_pi, obs, self.n_clones))

    def predict_next_obs(self, obs_seq):
        """Predict next observation from Viterbi last state.

        Returns
        -------
        pred_obs  : int
        log_probs : jax.Array (n_obs,)
        """
        states     = self.decode(obs_seq)
        last_state = int(states[-1])
        C          = self.n_clones
        # Reshape row of log_T into (n_obs, C), logsumexp over clones
        log_probs  = logsumexp(
            self.log_T[last_state].reshape(self.n_obs, C), axis=1
        )
        log_probs  = log_probs - logsumexp(log_probs)
        pred_obs   = int(jnp.argmax(log_probs))
        return pred_obs, log_probs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, sequences, n_iter=50, tol=1e-4, verbose=False,
            em_method="baum-welch"):
        """Baum-Welch EM (soft posteriors) or Viterbi EM (hard assignments).

        Parameters
        ----------
        em_method : str
            'baum-welch' (default) or 'viterbi'.

        Returns
        -------
        model : CSCG  (new instance)
        log_likelihoods : list[float]
        """
        if em_method == "viterbi":
            return self.fit_viterbi(sequences, n_iter=n_iter, tol=tol,
                                    verbose=verbose)
        model          = self
        log_likelihoods = []
        t_fit_start    = time.time()

        for iteration in range(n_iter):
            t_iter = time.time()
            total_ll, log_xi_sum, log_gamma0_sum = model._e_step(
                sequences, verbose=verbose
            )
            log_likelihoods.append(total_ll)
            if verbose:
                delta = (f"  Δ={total_ll - log_likelihoods[-2]:+.2f}"
                         if iteration > 0 else "")
                print(f"  BW iter {iteration:3d}: LL={total_ll:.2f}{delta}"
                      f"  ({time.time()-t_iter:.1f}s)")
            if iteration > 0 and total_ll - log_likelihoods[-2] < tol:
                if verbose:
                    print(f"  Converged after {iteration+1} iters "
                          f"({time.time()-t_fit_start:.1f}s total)")
                break
            model = model._m_step(log_xi_sum, log_gamma0_sum)

        return model, log_likelihoods

    def fit_viterbi(self, sequences, n_iter=50, tol=1e-4, verbose=False):
        """Viterbi EM (hard MAP assignments).

        Returns
        -------
        model : CSCG
        log_likelihoods : list[float]
        """
        model           = self
        log_likelihoods = []
        N, C            = self.n_states, self.n_clones

        for iteration in range(n_iter):
            if verbose:
                lengths = [len(s) for s in sequences]
                print(f"    [viterbi scan {len(sequences)}×{max(lengths)}t"
                      f"{'(padded)' if len(set(lengths))>1 else ''}] running...",
                      flush=True)
            t0 = time.time()
            obs_batch, true_lens = _pad_sequences(sequences)
            total_ll, log_xi_sum, log_gamma0_sum = _viterbi_e_step_scan(
                model.log_T, model.log_pi, obs_batch, true_lens, C, N
            )
            if verbose:
                print(f"    E-step done {time.time()-t0:.1f}s", flush=True)
            total_ll = float(total_ll)
            log_likelihoods.append(total_ll)
            if verbose:
                delta = (f"  Δ={total_ll - log_likelihoods[-2]:+.2f}"
                         if iteration > 0 else "")
                print(f"  Viterbi EM iter {iteration:3d}: LL={total_ll:.2f}{delta}")
            if iteration > 0 and total_ll - log_likelihoods[-2] < tol:
                break
            model = model._m_step(log_xi_sum, log_gamma0_sum)

        return model, log_likelihoods

    # ------------------------------------------------------------------
    # E-step and M-step
    # ------------------------------------------------------------------

    def _e_step(self, sequences, verbose=False):
        """JIT-compiled E-step: pad sequences and scan over them.

        Returns
        -------
        total_ll       : float
        log_xi_sum     : (N, N)
        log_gamma0_sum : (N,)
        """
        N, C = self.n_states, self.n_clones
        if verbose:
            lengths = [len(s) for s in sequences]
            print(f"    [scan {len(sequences)}×{max(lengths)}t"
                  f"{'(padded)' if len(set(lengths))>1 else ''}] running...",
                  flush=True)
        t0 = time.time()
        obs_batch, true_lens = _pad_sequences(sequences)
        total_ll, log_xi_sum, log_gamma0_sum = _e_step_scan(
            self.log_T, self.log_pi, obs_batch, true_lens, C, N
        )
        jax.block_until_ready(log_xi_sum)
        if verbose:
            print(f"    E-step done {time.time()-t0:.1f}s", flush=True)
        return float(total_ll), log_xi_sum, log_gamma0_sum

    def _m_step(self, log_xi_sum, log_gamma0_sum):
        """M-step: reestimate log_T and log_pi from sufficient statistics."""
        eps    = max(self.pseudocount, 1e-300)
        log_eps = jnp.log(jnp.array(eps))

        log_xi_smoothed = jnp.logaddexp(log_xi_sum, log_eps)
        row_sums        = logsumexp(log_xi_smoothed, axis=1, keepdims=True)
        log_T_new       = log_xi_smoothed - row_sums

        log_pi_new = _log_normalize(jnp.logaddexp(log_gamma0_sum, log_eps))

        return eqx.tree_at(
            lambda m: (m.log_T, m.log_pi),
            self,
            (log_T_new, log_pi_new),
        )
