#!/usr/bin/env python3
"""Financial time series experiment: CSCG vs SM-CSCG on ETF return regimes.

Discretizes daily ETF returns via argmax encoding (which ETF had the highest
return each day), producing a categorical observation sequence. Models are
evaluated using an expanding-window protocol: train on years [first, ..., Y-1],
test on year Y, incrementing Y.

Compares CSCG and SM-CSCG across clone counts and phase counts, measuring:
  1. BPS (bits per symbol) — compression / predictive quality
  2. Next-token prediction accuracy — can the model predict tomorrow's winner?

Plots: per-year BPS, per-year accuracy, latent state time series, rolling accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx

from smcscg import CSCG, SMCSCG


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ETFS = [
    "SPY", "QQQ", "XLF", "XLE", "XLV", "XLK", "XLU", "XLP",
    "XLI", "XLB", "XLY", "IWM", "EFA", "EEM", "VNQ", "GLD", "TLT", "BIL",
]


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def download_etf_data(tickers, start="2005-01-01", end=None, cache_dir="data/financial"):
    """Download daily adjusted close prices via yfinance, with parquet caching.

    Aligns all tickers to their common date range (starts from the latest
    inception date). Forward-fills any gaps. Drops tickers with >5% NaN.

    Returns
    -------
    prices : pd.DataFrame — (n_days, n_tickers) with DatetimeIndex
    """
    import yfinance as yf

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Cache key from sorted tickers + date range
    key_str = "_".join(sorted(tickers)) + f"_{start}_{end or 'now'}"
    cache_key = hashlib.md5(key_str.encode()).hexdigest()[:12]
    csv_file = cache_path / f"etf_prices_{cache_key}.csv"

    # Use cache if fresh (< 24 hours old)
    if csv_file.exists():
        age_hours = (time.time() - csv_file.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"  Loading cached prices from {csv_file}")
            prices = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            return prices

    print(f"  Downloading {len(tickers)} ETFs from yfinance...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False)

    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data  # single ticker

    # Align to common date range FIRST (start from latest inception date)
    first_valid = prices.apply(lambda s: s.first_valid_index())
    common_start = first_valid.max()
    prices = prices.loc[common_start:]

    # Now check for remaining NaNs (should be minimal after alignment)
    nan_frac = prices.isna().mean()
    bad_tickers = nan_frac[nan_frac > 0.05].index.tolist()
    if bad_tickers:
        print(f"  Warning: dropping tickers with >5% NaN after alignment: {bad_tickers}")
        prices = prices.drop(columns=bad_tickers)

    print(f"  Common start date: {common_start.date()} "
          f"({len(prices)} trading days, {len(prices.columns)} tickers)")

    # Forward-fill then backward-fill remaining gaps
    prices = prices.ffill().bfill()

    # Ensure column order matches input ticker order (for those that survived)
    surviving = [t for t in tickers if t in prices.columns]
    prices = prices[surviving]

    prices.to_csv(csv_file)
    print(f"  Cached to {csv_file}")
    return prices


def compute_log_returns(prices):
    """Compute daily log returns: log(P_t / P_{t-1}). Drops first NaN row."""
    return np.log(prices / prices.shift(1)).dropna()


def encode_observations(log_returns):
    """Encode daily returns as argmax observations.

    Each day, the observation is the index of the ETF with the highest
    log return. Returns int32 array in {0, ..., n_etfs-1} and ticker names.
    """
    obs_seq = np.argmax(log_returns.values, axis=1).astype(np.int32)
    return obs_seq, list(log_returns.columns)


def split_by_year(obs_seq, dates, min_train_years=5):
    """Create expanding-window train/test splits by calendar year.

    Returns a list of dicts with 'train_seq', 'test_seq', 'train_years',
    'test_year', 'train_dates', 'test_dates'.
    """
    years = sorted(dates.year.unique())
    splits = []

    for test_year in years[min_train_years:]:
        train_mask = dates.year < test_year
        test_mask = dates.year == test_year

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        splits.append({
            'train_seq': obs_seq[train_mask],
            'test_seq': obs_seq[test_mask],
            'train_years': (years[0], test_year - 1),
            'test_year': test_year,
            'train_dates': dates[train_mask],
            'test_dates': dates[test_mask],
        })

    return splits


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model_configs(n_obs, clone_counts, sm_phases, seed, pseudocount=1e-6,
                        mean_duration=None):
    """Build all CSCG and SM-CSCG model configurations.

    Returns list of dicts with 'name', 'model_type', 'n_clones', 'n_phases',
    and 'factory' (callable returning a fresh model instance).
    """
    configs = []

    for c in clone_counts:
        key_int = seed + c * 1000
        configs.append({
            'name': f"CSCG_C={c}",
            'model_type': 'cscg',
            'n_clones': c,
            'n_phases': 1,
            'factory': lambda n=n_obs, nc=c, k=key_int, pc=pseudocount: CSCG(
                n_obs=n, n_clones=nc, pseudocount=pc, key=jax.random.PRNGKey(k)),
        })

    for c in clone_counts:
        for l in sm_phases:
            key_int = seed + c * 1000 + l * 100
            configs.append({
                'name': f"SM_C={c}_L={l}",
                'model_type': 'smcscg',
                'n_clones': c,
                'n_phases': l,
                'factory': lambda n=n_obs, nc=c, nl=l, k=key_int, pc=pseudocount, md=mean_duration: SMCSCG(
                    n_obs=n, n_clones=nc, n_phases=nl, pseudocount=pc,
                    mean_duration=md, key=jax.random.PRNGKey(k)),
            })

    return configs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_prediction_accuracy(model, obs_seq):
    """Compute next-token prediction accuracy over a sequence.

    predict_next_obs(obs_seq) returns (T, n_obs) where log_probs[t] gives
    P(x_{t+1} | x_{1:t}). We compare argmax predictions to true next tokens.

    Returns overall accuracy, transition accuracy (only at regime changes),
    and mean log-probability of the true next token.

    Returns
    -------
    accuracy, transition_accuracy, mean_log_prob : float, float, float
    """
    log_probs = model.predict_next_obs(obs_seq)  # (T, n_obs)
    # log_probs[t] predicts obs at t+1, so compare [:-1] with obs[1:]
    predictions = jnp.argmax(log_probs[:-1], axis=1)
    true_next = jnp.asarray(obs_seq[1:], dtype=jnp.int32)
    correct = predictions == true_next
    accuracy = float(correct.mean())

    # Transition accuracy: only at points where obs changes
    current = jnp.asarray(obs_seq[:-1], dtype=jnp.int32)
    transition_mask = current != true_next
    n_transitions = int(transition_mask.sum())
    if n_transitions > 0:
        transition_accuracy = float(correct[transition_mask].mean())
    else:
        transition_accuracy = float('nan')

    # Mean log-prob of true next token (proper scoring rule)
    # log_probs[:-1] has T-1 rows predicting obs[1:T], true_next has T-1 elements
    true_log_probs = log_probs[jnp.arange(len(true_next)), true_next]
    mean_log_prob = float(true_log_probs.mean())

    return accuracy, transition_accuracy, mean_log_prob


def compute_rolling_accuracy(model, obs_seq, window_sizes=(21, 63)):
    """Compute rolling prediction accuracy with specified window sizes.

    Returns dict mapping window_size -> rolling accuracy array of length T-1.
    """
    log_probs = model.predict_next_obs(obs_seq)
    predictions = jnp.argmax(log_probs[:-1], axis=1)
    true_next = jnp.asarray(obs_seq[1:], dtype=jnp.int32)
    correct = pd.Series(np.array(predictions == true_next, dtype=np.float64))

    return {w: correct.rolling(w, min_periods=1).mean().values
            for w in window_sizes}


def run_expanding_window(obs_seq, dates, model_configs, n_iter, tol,
                         min_train_years, verbose):
    """Run expanding-window evaluation for all model configurations.

    Returns list of result dicts.
    """
    splits = split_by_year(obs_seq, dates, min_train_years)
    results = []

    for split in splits:
        test_year = split['test_year']
        n_train = len(split['train_seq'])
        n_test = len(split['test_seq'])
        print(f"\n{'='*70}")
        print(f"Test year: {test_year}  "
              f"(train: {split['train_years'][0]}-{split['train_years'][1]}, "
              f"{n_train} days | test: {n_test} days)")
        print(f"{'='*70}")

        for cfg in model_configs:
            t0 = time.time()
            model = cfg['factory']()
            model, lls = model.fit(
                [split['train_seq']],
                n_iter=n_iter,
                tol=tol,
                verbose=verbose,
                em_method="baum-welch",
            )
            elapsed = time.time() - t0

            test_bps = float(model.bps(split['test_seq']))
            pred_acc, trans_acc, mean_lp = compute_prediction_accuracy(
                model, split['test_seq'])

            results.append({
                'config_name': cfg['name'],
                'model_type': cfg['model_type'],
                'n_clones': cfg['n_clones'],
                'n_phases': cfg['n_phases'],
                'test_year': test_year,
                'n_train_days': n_train,
                'n_test_days': n_test,
                'test_bps': test_bps,
                'prediction_accuracy': pred_acc,
                'transition_accuracy': trans_acc,
                'mean_log_prob': mean_lp,
                'n_iters': len(lls),
                'elapsed_s': elapsed,
            })

            print(f"  {cfg['name']:<20s} ({len(lls):>3} it, {elapsed:>5.1f}s)  "
                  f"BPS={test_bps:.4f}  Acc={pred_acc:.4f}  "
                  f"TransAcc={trans_acc:.4f}  MLP={mean_lp:.4f}")

    return results


# ---------------------------------------------------------------------------
# Latent state analysis
# ---------------------------------------------------------------------------

def train_and_decode_best(obs_seq, dates, best_cfg, n_iter, tol, verbose):
    """Train the best model on full data and decode latent states.

    Returns (model, latent_states).
    """
    print(f"\nTraining best model ({best_cfg['name']}) on full dataset...")
    model = best_cfg['factory']()
    model, lls = model.fit([obs_seq], n_iter=n_iter, tol=tol, verbose=verbose,
                           em_method="baum-welch")
    print(f"  Converged in {len(lls)} iterations")

    if best_cfg['model_type'] == 'smcscg':
        macro_states, _ = model.decode(obs_seq)
        latent_states = np.array(macro_states)
    else:
        latent_states = np.array(model.decode(obs_seq))

    return model, latent_states


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_data_summary(obs_seq, ticker_names):
    """Print observation frequency table."""
    n_obs = len(ticker_names)
    counts = np.bincount(obs_seq, minlength=n_obs)
    total = len(obs_seq)
    uniform_bps = np.log2(n_obs)

    # Empirical entropy
    freqs = counts / total
    freqs_nz = freqs[freqs > 0]
    empirical_bps = -np.sum(freqs_nz * np.log2(freqs_nz))

    print(f"\nObservation frequency (which ETF had highest return):")
    # Sort by frequency descending
    order = np.argsort(-counts)
    for i in order:
        print(f"  {ticker_names[i]:>5s}: {counts[i]:>5d} days ({100*counts[i]/total:>5.1f}%)")
    print(f"\n  Uniform BPS baseline:  {uniform_bps:.4f}")
    print(f"  Empirical entropy BPS: {empirical_bps:.4f}")
    print(f"  Chance accuracy:       {1/n_obs:.4f}")


def print_results_table(results, ticker_names):
    """Print per-year and aggregate results tables."""
    df = pd.DataFrame(results)
    n_obs = len(ticker_names)
    uniform_bps = np.log2(n_obs)
    chance_acc = 1.0 / n_obs

    w = 105
    header = (f"{'Config':<20s} {'Year':>4} {'Train':>5} {'Test':>4} "
              f"{'BPS':>8} {'Acc':>6} {'TransAcc':>8} {'MLP':>8} "
              f"{'Iters':>5} {'Time':>6}")
    print(f"\n{'='*w}")
    print(f"RESULTS  (Uniform BPS = {uniform_bps:.4f}, Chance Acc = {chance_acc:.4f})")
    print("=" * w)
    print(header)
    print("-" * w)

    for _, r in df.iterrows():
        print(f"{r['config_name']:<20s} {r['test_year']:>4d} "
              f"{r['n_train_days']:>5d} {r['n_test_days']:>4d} "
              f"{r['test_bps']:>8.4f} {r['prediction_accuracy']:>6.4f} "
              f"{r['transition_accuracy']:>8.4f} {r['mean_log_prob']:>8.4f} "
              f"{r['n_iters']:>5d} {r['elapsed_s']:>5.1f}s")

    # Aggregate summary
    print(f"\n{'='*w}")
    print("AGGREGATE (mean +/- std over test years)")
    print("=" * w)
    agg_header = (f"{'Config':<20s} {'BPS':>14} {'TransAcc':>14} "
                  f"{'MLP':>14}")
    print(agg_header)
    print("-" * w)

    for name in df['config_name'].unique():
        sub = df[df['config_name'] == name]
        bps_mean, bps_std = sub['test_bps'].mean(), sub['test_bps'].std()
        tacc_mean, tacc_std = sub['transition_accuracy'].mean(), sub['transition_accuracy'].std()
        mlp_mean, mlp_std = sub['mean_log_prob'].mean(), sub['mean_log_prob'].std()
        print(f"{name:<20s} {bps_mean:>6.4f} +/- {bps_std:<6.4f} "
              f"{tacc_mean:>6.4f} +/- {tacc_std:<6.4f} "
              f"{mlp_mean:>6.4f} +/- {mlp_std:<6.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, ticker_names, best_model, best_latent_states,
                 full_obs_seq, full_dates, best_cfg, save_dir):
    """Generate and save all 4 plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plots)")
        return

    os.makedirs(save_dir, exist_ok=True)

    plot_bps_by_year(results, ticker_names, save_dir)
    plot_accuracy_by_year(results, ticker_names, save_dir)
    plot_latent_states(best_latent_states, full_dates, full_obs_seq,
                       ticker_names, best_cfg['name'], save_dir)
    plot_rolling_accuracy(best_model, full_obs_seq, full_dates,
                          ticker_names, save_dir)


def _style_config_lines(df, ax, metric_col):
    """Plot lines for each model config with consistent styling.

    CSCG = solid lines, SM-CSCG = dashed.
    Color by n_clones, different markers for n_phases.
    """
    import matplotlib.pyplot as plt

    clone_colors = {1: 'C0', 2: 'C1', 4: 'C2', 8: 'C3', 16: 'C4'}
    phase_markers = {1: 'o', 2: 's', 4: '^', 6: 'D'}

    for name in df['config_name'].unique():
        sub = df[df['config_name'] == name].sort_values('test_year')
        row = sub.iloc[0]
        c = row['n_clones']
        l = row['n_phases']
        ls = '-' if row['model_type'] == 'cscg' else '--'
        color = clone_colors.get(c, 'grey')
        marker = phase_markers.get(l, 'x')
        ax.plot(sub['test_year'], sub[metric_col],
                marker=marker, linestyle=ls, color=color,
                label=name, markersize=4, linewidth=1.2)


def plot_bps_by_year(results, ticker_names, save_dir):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    n_obs = len(ticker_names)
    uniform_bps = np.log2(n_obs)

    fig, ax = plt.subplots(figsize=(12, 6))
    _style_config_lines(df, ax, 'test_bps')
    ax.axhline(uniform_bps, color='grey', linestyle=':', alpha=0.6,
               label=f'Uniform ({uniform_bps:.2f})')
    ax.set(xlabel='Test Year', ylabel='BPS (bits per symbol)',
           title='Bits per Symbol by Test Year (lower = better)')
    ax.legend(fontsize=7, ncol=3, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "bps_by_year.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_accuracy_by_year(results, ticker_names, save_dir):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    n_obs = len(ticker_names)
    chance_acc = 1.0 / n_obs

    fig, ax = plt.subplots(figsize=(12, 6))
    _style_config_lines(df, ax, 'prediction_accuracy')
    ax.axhline(chance_acc, color='grey', linestyle=':', alpha=0.6,
               label=f'Chance ({chance_acc:.4f})')
    ax.set(xlabel='Test Year', ylabel='Prediction Accuracy',
           title='Next-Token Prediction Accuracy by Test Year (higher = better)')
    ax.legend(fontsize=7, ncol=3, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "accuracy_by_year.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_latent_states(latent_states, dates, obs_seq, ticker_names, model_name,
                       save_dir):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Top: latent states
    ax1.scatter(dates, latent_states, s=0.5, alpha=0.5, c=latent_states,
                cmap='tab20', rasterized=True)
    ax1.set(ylabel='Latent State', title=f'Latent States — {model_name}')
    ax1.grid(True, alpha=0.2)

    # Bottom: observations (winning ETF)
    ax2.scatter(dates, obs_seq, s=0.5, alpha=0.5, c=obs_seq,
                cmap='tab20', rasterized=True)
    ax2.set(xlabel='Date', ylabel='Winning ETF')
    ax2.set_yticks(range(len(ticker_names)))
    ax2.set_yticklabels(ticker_names, fontsize=6)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, "latent_states.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_rolling_accuracy(model, obs_seq, dates, ticker_names, save_dir):
    import matplotlib.pyplot as plt

    n_obs = len(ticker_names)
    chance_acc = 1.0 / n_obs

    rolling = compute_rolling_accuracy(model, obs_seq, window_sizes=(21, 63))
    # Dates for predictions: dates[1:] (predicting t+1 from t)
    pred_dates = dates[1:]

    fig, ax = plt.subplots(figsize=(16, 5))
    for w, vals in sorted(rolling.items()):
        ax.plot(pred_dates, vals, linewidth=0.8, alpha=0.8,
                label=f'{w}-day rolling accuracy')

    ax.axhline(chance_acc, color='grey', linestyle=':', alpha=0.6,
               label=f'Chance ({chance_acc:.4f})')
    ax.set(xlabel='Date', ylabel='Rolling Accuracy',
           title='Rolling Prediction Accuracy (best model, full dataset)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "rolling_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    t_start = time.time()

    # --- 1. Download and preprocess data ---
    print("\n--- Data Pipeline ---")
    prices = download_etf_data(args.tickers, start=args.start_date,
                               cache_dir=args.cache_dir)
    log_returns = compute_log_returns(prices)
    if args.smooth_window > 1:
        log_returns = log_returns.rolling(args.smooth_window, min_periods=1).mean()
    obs_seq, ticker_names = encode_observations(log_returns)
    dates = log_returns.index
    n_obs = len(ticker_names)

    print(f"\n  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  {len(obs_seq)} trading days, {n_obs} ETFs")

    print_data_summary(obs_seq, ticker_names)

    # --- 2. Build model configs ---
    model_configs = build_model_configs(
        n_obs=n_obs,
        clone_counts=args.clone_counts,
        sm_phases=args.sm_phases,
        seed=args.seed,
        pseudocount=args.pseudocount,
        mean_duration=args.mean_duration,
    )
    print(f"\n--- Models: {len(model_configs)} configurations ---")
    for cfg in model_configs:
        print(f"  {cfg['name']}")

    # --- 3. Expanding-window evaluation ---
    print(f"\n--- Expanding-Window Evaluation (min_train_years={args.min_train_years}) ---")
    results = run_expanding_window(
        obs_seq, dates, model_configs,
        n_iter=args.n_iter, tol=args.tol,
        min_train_years=args.min_train_years,
        verbose=args.verbose,
    )

    # --- 4. Print results ---
    print_results_table(results, ticker_names)

    # --- 5. Train best model on full data, decode latent states ---
    df = pd.DataFrame(results)
    agg = df.groupby('config_name').agg(
        mean_bps=('test_bps', 'mean'),
        mean_acc=('prediction_accuracy', 'mean'),
    )
    best_name = agg['mean_bps'].idxmin()
    best_cfg = next(c for c in model_configs if c['name'] == best_name)
    print(f"\n  Best model by mean BPS: {best_name} "
          f"(BPS={agg.loc[best_name, 'mean_bps']:.4f}, "
          f"Acc={agg.loc[best_name, 'mean_acc']:.4f})")

    best_model, best_latent = train_and_decode_best(
        obs_seq, dates, best_cfg, n_iter=args.n_iter, tol=args.tol,
        verbose=args.verbose,
    )

    # --- 6. Plots ---
    if args.plot:
        print(f"\n--- Generating Plots ---")
        plot_results(results, ticker_names, best_model, best_latent,
                     obs_seq, dates, best_cfg, args.plot)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Financial time series experiment: CSCG vs SM-CSCG"
    )
    parser.add_argument("--platform", default="cpu",
                        choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--tickers", type=str, nargs="+", default=DEFAULT_ETFS,
                        help="ETF tickers to use")
    parser.add_argument("--start_date", type=str, default="2005-01-01",
                        help="Start date for price data download")
    parser.add_argument("--cache_dir", type=str, default="data/financial",
                        help="Directory to cache downloaded price data")
    parser.add_argument("--smooth_window", type=int, default=1,
                        help="Rolling mean window for log returns before argmax (1=no smoothing)")

    # Models
    parser.add_argument("--clone_counts", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--sm_phases", type=int, nargs="+",
                        default=[2, 5])
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--pseudocount", type=float, default=1.0,
                        help="Laplace smoothing for sufficient statistics")
    parser.add_argument("--mean_duration", type=float, default=None,
                        help="Target mean duration for SM-CSCG phase-type init")

    # Evaluation
    parser.add_argument("--min_train_years", type=int, default=5,
                        help="Minimum years of training data before first test")

    # Output
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", type=str, default=None,
                        help="Directory to save plots (e.g., results/financial)")

    # Quick mode
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: fewer configs, fewer EM iters")

    args = parser.parse_args()

    jax.config.update("jax_platform_name", args.platform)

    if args.quick:
        args.tickers = ["SPY", "QQQ", "XLF", "XLE", "GLD", "TLT", "BIL"]
        args.clone_counts = [1, 2, 4]
        args.sm_phases = [2, 4]
        args.n_iter = 30
        args.min_train_years = 10
        args.smooth_window = 10

    print("Financial Time Series Experiment: CSCG vs SM-CSCG")
    print(f"  Tickers:     {args.tickers}")
    print(f"  Smooth window: {args.smooth_window}")
    print(f"  Clone counts: {args.clone_counts}")
    print(f"  SM phases:    {args.sm_phases}")
    print(f"  EM iters:     {args.n_iter}, tol={args.tol}, pseudocount={args.pseudocount}")
    print(f"  Mean duration: {args.mean_duration}")
    print(f"  Min train years: {args.min_train_years}")

    run_experiment(args)


if __name__ == "__main__":
    main()
