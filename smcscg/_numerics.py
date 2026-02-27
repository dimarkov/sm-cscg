"""Shared numerical utilities for log-space HMM/HSMM computations."""

import numpy as np
from scipy.special import logsumexp

NEG_INF = -np.inf


def log_normalize(log_vec):
    """Normalize a log-probability vector so exp(result).sum() == 1."""
    return log_vec - logsumexp(log_vec)


def log_matmul(log_A, log_B):
    """Matrix multiplication in log-space: log(exp(A) @ exp(B)).

    Parameters
    ----------
    log_A : ndarray, shape (M, K)
    log_B : ndarray, shape (K, N)

    Returns
    -------
    ndarray, shape (M, N)
    """
    M, K = log_A.shape
    _, N = log_B.shape
    # log_A[:, :, None] + log_B[None, :, :] -> (M, K, N)
    return logsumexp(log_A[:, :, None] + log_B[None, :, :], axis=1)


def build_clone_emission_matrix(n_obs, n_clones):
    """Build deterministic sparse emission matrix for CSCG.

    Clone c_{o,k} (hidden state o*n_clones + k) emits observation o
    with probability 1 and all others with probability 0.

    Parameters
    ----------
    n_obs : int
        Number of unique observations.
    n_clones : int
        Number of clones per observation.

    Returns
    -------
    log_E : ndarray, shape (n_states, n_obs)
        Log emission matrix.  0.0 where clone emits, -inf elsewhere.
    clone_to_obs : ndarray, shape (n_states,)
        Maps each hidden state index to its observation index.
    """
    n_states = n_obs * n_clones
    log_E = np.full((n_states, n_obs), NEG_INF)
    clone_to_obs = np.empty(n_states, dtype=np.intp)
    for o in range(n_obs):
        start = o * n_clones
        end = start + n_clones
        log_E[start:end, o] = 0.0
        clone_to_obs[start:end] = o
    return log_E, clone_to_obs


def clones_for_obs(clone_to_obs, n_obs):
    """Precompute mapping from observation -> array of clone indices.

    Returns
    -------
    obs_to_clones : list of ndarray
        obs_to_clones[o] is a 1-D int array of hidden state indices
        that emit observation o.
    """
    obs_to_clones = []
    for o in range(n_obs):
        obs_to_clones.append(np.where(clone_to_obs == o)[0])
    return obs_to_clones


def build_expanded_emission_matrix(n_obs, n_clones, n_phases):
    """Build emission matrix for phase-type expanded state space.

    Expanded state (j, l) = clone j, phase l.
    Linear index: j * n_phases + l.
    All L phases of clone j emit obs(j) deterministically.

    Parameters
    ----------
    n_obs, n_clones, n_phases : int

    Returns
    -------
    log_E : ndarray, shape (n_obs * n_clones * n_phases, n_obs)
    clone_to_obs : ndarray, shape (n_obs * n_clones * n_phases,)
        Maps each expanded state to its observation index.
    """
    N = n_obs * n_clones
    N_tilde = N * n_phases
    log_E = np.full((N_tilde, n_obs), NEG_INF)
    clone_to_obs = np.empty(N_tilde, dtype=np.intp)
    for o in range(n_obs):
        for k in range(n_clones):
            j = o * n_clones + k  # macro-state index
            for l in range(n_phases):
                idx = j * n_phases + l
                log_E[idx, o] = 0.0
                clone_to_obs[idx] = o
    return log_E, clone_to_obs


def precompute_run_lengths(obs_seq, n_obs):
    """For each position t and observation o, compute the max contiguous
    run of o starting at t.

    Parameters
    ----------
    obs_seq : ndarray, shape (T,)
    n_obs : int

    Returns
    -------
    max_run : ndarray, shape (T, n_obs)
        max_run[t, o] = length of contiguous run of o starting at t
        (0 if obs_seq[t] != o).
    """
    T = len(obs_seq)
    max_run = np.zeros((T, n_obs), dtype=np.intp)
    for o in range(n_obs):
        run = 0
        for t in range(T - 1, -1, -1):
            if obs_seq[t] == o:
                run += 1
            else:
                run = 0
            max_run[t, o] = run
    return max_run
