"""GINC data generation and loading for CSCG experiments.

Wraps the original GINC generator from:
  https://github.com/p-lambda/incontext-learning

Generates data using the original HMM structure (n_values × n_slots hidden
states, n_symbols vocabulary) and returns integer sequences for CSCG training.

Parameters matching the GINC paper defaults:
  transition_temp=0.1, start_temp=10.0, n_symbols=50, n_values=10,
  n_slots=10, value_identity_coeff=0.9, n_hmms=10

Usage
-----
    hmm = GINCDataset(data_dir='data/ginc', n_hmms=10, n_symbols=50,
                      n_values=10, n_slots=10)
    train_seqs = hmm.get_train_sequences()
    prompts, labels = hmm.get_test_prompts(sentence_len=8, n_context=8)
"""

import json
import pickle
import argparse
import sys
import os
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal copy of generate_data.py data-generation logic (no HuggingFace)
# ---------------------------------------------------------------------------

from contextlib import contextmanager
from functools import partial
from string import ascii_lowercase
from itertools import permutations

try:
    from hmmlearn.hmm import CategoricalHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


def _softmax(x, temp=1.0, axis=None):
    x = x / temp
    if axis is None:
        x -= np.amax(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x -= np.expand_dims(np.amax(x, axis=axis), axis=axis)
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis=axis)


def _generate_transmat_block(n_components, perm_samples=10, transition_temp=1.0):
    mixing = _softmax(np.random.rand(perm_samples) - 0.5, transition_temp)
    mixing = mixing[:, np.newaxis, np.newaxis]
    perms = [np.eye(n_components)[np.random.permutation(n_components)]
             for _ in range(perm_samples)]
    return np.sum(mixing * perms, axis=0)


def _combine_transmats(mat1, mat2):
    n, m = mat1.shape[0], mat2.shape[0]
    mat = np.zeros((m * n, m * n))
    for i in range(m):
        for j in range(m):
            mat[i*n:(i+1)*n, j*n:(j+1)*n] = mat1 * mat2[i, j]
    return mat


@contextmanager
def _local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _letter_generator(num):
    counter = 0
    for i in range(1, len(ascii_lowercase)):
        for perm in permutations(ascii_lowercase, i):
            yield ''.join(perm)
            counter += 1
            if counter >= num:
                return


def _generate_hmm_params(n_values, n_slots, n_symbols, all_values,
                          perm_samples=10, transition_temp=1.0, start_temp=1.0,
                          value_id_coeff=0.8, value_seed=1112):
    n_components = n_values * n_slots
    startprob = _softmax(np.random.rand(n_components) - 0.5, start_temp)
    slot_transmat = _generate_transmat_block(
        n_slots, perm_samples=n_slots, transition_temp=transition_temp)
    with _local_seed(value_seed):
        value_transmat = _generate_transmat_block(
            n_values, perm_samples=n_values, transition_temp=transition_temp)
        value_transmat = ((1 - value_id_coeff) * value_transmat
                          + value_id_coeff * np.eye(n_values))
    transmat = _combine_transmats(slot_transmat, value_transmat)
    emissionprob = np.zeros((n_components, n_symbols))
    for i in range(n_components):
        slot_idx = i % n_slots
        value_idx = i // n_slots
        emissionprob[i, all_values[value_idx, slot_idx]] = 1
    return startprob, transmat, emissionprob, slot_transmat, value_transmat


def _sample_from_hmm(hmm, length, seed=None):
    x, h = hmm.sample(n_samples=length, random_state=seed)
    return x.T[0], h


def _generate_hiddens_from_state(hmm, start_state, length):
    hiddens = [start_state]
    for _ in range(length):
        hiddens.append(
            np.random.choice(hmm.transmat_.shape[1],
                             p=hmm.transmat_[hiddens[-1], :]))
    return hiddens


def _score_hmm(hmm, prompt, start_dist=None):
    if start_dist is not None:
        old_sp = hmm.startprob_
        hmm.startprob_ = start_dist
    prompt_arr = np.asarray(prompt).reshape(-1, 1)
    proba = hmm.predict_proba(prompt_arr)
    proba_next_hidden = hmm.transmat_.T @ proba[-1]
    proba_next_emission = hmm.emissionprob_.T @ proba_next_hidden
    if start_dist is not None:
        hmm.startprob_ = old_sp
    return proba_next_emission


# ---------------------------------------------------------------------------
# GINCDataset class
# ---------------------------------------------------------------------------

class GINCDataset:
    """GINC dataset generator / loader.

    Parameters
    ----------
    data_dir    : str or Path — directory to store/load generated data
    n_symbols   : int — vocabulary size (excludes delimiter '/')
    n_values    : int — number of entity types
    n_slots     : int — number of property slots
    n_hmms      : int — total HMMs (n_id = n_hmms // 2)
    transition_temp, start_temp, value_id_coeff : float — HMM generation params
    seed        : int — random seed for reproducibility
    """

    def __init__(self, data_dir='data/ginc', n_symbols=50, n_values=10,
                 n_slots=10, n_hmms=10, transition_temp=0.1, start_temp=10.0,
                 value_id_coeff=0.9, seed=1111):
        assert _HMM_AVAILABLE, "hmmlearn is required: uv add hmmlearn (needs CategoricalHMM)"
        self.data_dir = Path(data_dir)
        self.n_symbols = n_symbols
        self.n_values = n_values
        self.n_slots = n_slots
        self.n_hmms = n_hmms
        self.n_id_hmms = n_hmms // 2
        self.transition_temp = transition_temp
        self.start_temp = start_temp
        self.value_id_coeff = value_id_coeff
        self.seed = seed
        self.n_components = n_values * n_slots

        # Build vocabulary (index 0 = '/' delimiter, indices 1..n_symbols = tokens)
        vocab_list = list(_letter_generator(n_symbols))
        vocab_list = ['/'] + vocab_list[:-1]   # replace last with '/'
        self.vocab = np.asarray(vocab_list)
        self.vocab_to_int = {v: i for i, v in enumerate(self.vocab)}

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_or_generate()

    def _load_or_generate(self):
        hmm_path = self.data_dir / 'id_hmms.pkl'
        if hmm_path.exists():
            self._load_existing()
        else:
            self._generate()

    def _generate(self):
        """Generate HMMs and training/validation data."""
        print(f"Generating GINC data in {self.data_dir} ...")
        np.random.seed(self.seed)

        all_values = np.random.randint(low=1, high=len(self.vocab),
                                       size=(self.n_values, self.n_slots))
        all_values[:, 0] = 0  # delimiter column
        self.all_values = all_values

        hmm_list = []
        hmm_params = []
        for _ in range(self.n_hmms):
            sp, tm, em, st, vt = _generate_hmm_params(
                self.n_values, self.n_slots, self.n_symbols, all_values,
                perm_samples=self.n_components,
                transition_temp=self.transition_temp,
                start_temp=self.start_temp,
                value_id_coeff=self.value_id_coeff,
                value_seed=self.seed + 3,
            )
            hmm = CategoricalHMM(n_components=self.n_components)
            hmm.startprob_ = sp
            hmm.transmat_ = tm
            hmm.emissionprob_ = em
            hmm_list.append(hmm)
            hmm_params.append((st, vt))

        self.id_hmms = hmm_list[:self.n_id_hmms]
        self.id_params = hmm_params[:self.n_id_hmms]
        self.ood_hmms = hmm_list[self.n_id_hmms:]

        with open(self.data_dir / 'id_hmms.pkl', 'wb') as f:
            pickle.dump(self.id_hmms, f)
        with open(self.data_dir / 'id_params.pkl', 'wb') as f:
            pickle.dump(self.id_params, f)
        with open(self.data_dir / 'ood_hmms.pkl', 'wb') as f:
            pickle.dump(self.ood_hmms, f)
        with open(self.data_dir / 'all_values.pkl', 'wb') as f:
            pickle.dump(all_values, f)
        print(f"  Generated {len(self.id_hmms)} ID HMMs, {len(self.ood_hmms)} OOD HMMs")

    def _load_existing(self):
        """Load previously generated HMMs."""
        with open(self.data_dir / 'id_hmms.pkl', 'rb') as f:
            self.id_hmms = pickle.load(f)
        with open(self.data_dir / 'id_params.pkl', 'rb') as f:
            self.id_params = pickle.load(f)
        with open(self.data_dir / 'ood_hmms.pkl', 'rb') as f:
            self.ood_hmms = pickle.load(f)
        with open(self.data_dir / 'all_values.pkl', 'rb') as f:
            self.all_values = pickle.load(f)
        print(f"  Loaded GINC from {self.data_dir} "
              f"({len(self.id_hmms)} ID HMMs)")

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def get_train_sequences(self, n_docs=100, sample_length=10240, seed=None):
        """Sample training documents from ID HMMs.

        Parameters
        ----------
        n_docs        : int — documents per concept (HMM)
        sample_length : int — tokens per document
        seed          : int or None

        Returns
        -------
        sequences : list of ndarray (int32)  — integer token sequences
        labels    : list of int              — HMM index for each doc
        """
        rng_state = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)

        sequences = []
        labels = []
        for j, hmm in enumerate(self.id_hmms):
            for _ in range(n_docs):
                x, _ = _sample_from_hmm(hmm, sample_length)
                sequences.append(np.array(x, dtype=np.int32))
                labels.append(j)

        if seed is not None:
            np.random.set_state(rng_state)
        return sequences, labels

    def get_test_prompts(self, sentence_len=8, n_context_sentences=8,
                         n_prompts=2500, seed=1114):
        """Generate test prompts (context + next-token prediction task).

        Mirrors the 'ID_sample' prompt generation from the original repo.
        Each prompt = n_context complete examples + (sentence_len-1) tokens
        of a new example; label = the sentence_len-th token.

        Parameters
        ----------
        sentence_len        : int — k in paper (tokens per example)
        n_context_sentences : int — n in paper (number of context examples)
        n_prompts           : int
        seed                : int

        Returns
        -------
        prompts     : list of ndarray (int32)
        true_labels : list of int
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)

        prompts = []
        true_labels = []
        n_slots = self.n_slots
        n_values = self.n_values

        for _ in range(n_prompts):
            hmm_idx = np.random.choice(len(self.id_hmms))
            hmm = self.id_hmms[hmm_idx]

            # Choose a random start slot (>0 to avoid delimiter)
            start_slot = np.random.randint(low=1, high=n_slots)

            slots = []
            values = []
            for j in range(n_context_sentences + 1):
                start_value = np.random.randint(low=0, high=n_values)
                start_hidden = start_value * n_slots + start_slot
                h = _generate_hiddens_from_state(hmm, start_hidden,
                                                  length=sentence_len - 1)
                cur_slots = [hi % n_slots for hi in h]
                cur_values = [hi // n_slots for hi in h]
                slots += cur_slots
                values += cur_values
                slots += [0]          # delimiter
                values += [values[-1]]

            prompt_tokens = [self.all_values[values[j], slots[j]]
                             for j in range(len(slots))]
            # Remove final delimiter
            prompt_tokens = prompt_tokens[:-1]

            # Score the true next token
            x_prompt = prompt_tokens[-(sentence_len):-1]
            start_dist = np.zeros(hmm.startprob_.shape)
            for c in range(n_values):
                start_dist[c * n_slots + start_slot] = 1
            start_dist /= start_dist.sum()
            proba = _score_hmm(hmm, x_prompt, start_dist=start_dist)
            true_next = int(np.argmax(proba))

            # Prompt = all tokens except the last one (true label)
            prompt_seq = np.array(prompt_tokens[:-1], dtype=np.int32)
            prompts.append(prompt_seq)
            true_labels.append(true_next)

        np.random.set_state(rng_state)
        return prompts, true_labels


if __name__ == "__main__":
    ds = GINCDataset(data_dir='data/ginc_test', n_symbols=50, n_values=10,
                     n_slots=10, n_hmms=10)
    seqs, labels = ds.get_train_sequences(n_docs=2, sample_length=100, seed=42)
    print(f"Train: {len(seqs)} docs, lengths {[len(s) for s in seqs[:4]]}")
    print(f"Labels: {labels[:4]}")
    print(f"Sample tokens (first 20): {seqs[0][:20]}")
    print(f"Vocab size: {ds.n_symbols}")

    prompts, true_labels = ds.get_test_prompts(
        sentence_len=8, n_context_sentences=2, n_prompts=5, seed=42)
    print(f"\nTest prompts: {len(prompts)}, prompt len: {len(prompts[0])}")
    print(f"True labels: {true_labels}")
