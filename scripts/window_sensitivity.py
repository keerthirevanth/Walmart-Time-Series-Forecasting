"""Check that the backtest conclusions are not an artefact of the window count.

Re-runs the baseline set at several values of `n_backtest_windows` and reports
whether the WRMSSE ranking (and rough magnitude) is stable. If the ordering of
models is preserved across window counts, the default of 3 is defensible on
evidence rather than convention.

Usage:
    python -m scripts.window_sensitivity --windows 3 5 6
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.load import load_wide
from src.evaluation.backtest import run_backtest
from src.models.baselines import default_baselines


def _spearman(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Spearman rank correlation between two rank vectors."""
    a = pd.Series(rank_a).rank().to_numpy()
    b = pd.Series(rank_b).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[3, 5, 6],
        help="Window counts to compare.",
    )
    args = parser.parse_args()

    cfg = load_config()
    horizon = int(cfg.get("data.horizon"))
    sales_wide, calendar, prices = load_wide(cfg)
    print(f"Series: {sales_wide['id'].nunique():,} | horizon: {horizon}")

    records = {}
    for n_windows in args.windows:
        print(f"\n=== n_backtest_windows = {n_windows} ===")
        col = {}
        for forecaster in default_baselines():
            summary, _ = run_backtest(
                forecaster, sales_wide, calendar, prices, horizon, n_windows
            )
            wrmsse = summary["wrmsse"].mean()
            col[forecaster.name] = wrmsse
            print(f"  {forecaster.name:16s} WRMSSE={wrmsse:.4f}")
        records[n_windows] = col

    table = pd.DataFrame(records)
    table.columns = [f"w={w}" for w in args.windows]
    table["best_to_worst_rank"] = table[table.columns[0]].rank().astype(int)
    table = table.sort_values(table.columns[0])

    print("\nWRMSSE by window count (rows sorted best-first at the smallest count):\n")
    print(table.round(4).to_string())

    # Two separate questions matter here, and they are not the same:
    #  1. Is the BEST model the same regardless of window count? That is the
    #     conclusion we act on, so it is the one that must hold.
    #  2. Is the full ranking bit-identical? Useful context, but a swap between
    #     two near-tied mid-pack models is not a reason to distrust the setup.
    metric_cols = [f"w={w}" for w in args.windows]
    best_models = {c: table[c].idxmin() for c in metric_cols}
    best_stable = len(set(best_models.values())) == 1

    base = table[metric_cols[0]].to_numpy()
    print("\nRank correlation vs the smallest window count:")
    for w_col in metric_cols[1:]:
        rho = _spearman(base, table[w_col].to_numpy())
        print(f"  {metric_cols[0]} vs {w_col}: Spearman rho = {rho:.3f}")

    print(f"\nBest model at each window count: {best_models}")
    if best_stable:
        print(
            "Verdict: the best model is invariant to the window count, so the ranking "
            "conclusion is robust. Absolute WRMSSE rises with more windows because "
            "older origins cover harder periods; report final numbers at the larger "
            "count for a conservative estimate."
        )
    else:
        print(
            "Verdict: the best model changes with the window count. The evaluation is "
            "sensitive; prefer the larger window count and treat rankings with caution."
        )

    out = cfg.path("paths.backtests_dir")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "window_sensitivity.csv")
    print(f"Written to {out / 'window_sensitivity.csv'}")


if __name__ == "__main__":
    main()
