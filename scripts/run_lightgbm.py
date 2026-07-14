"""Tune and backtest the global LightGBM forecaster, then compare to the baselines.

Steps:
1. Optionally run an Optuna search (WRMSSE objective) on a held-out window.
2. Backtest the tuned model with the shared rolling-origin harness.
3. Write results and a feature-importance table, and print a combined leaderboard
   against the baseline numbers.

Usage:
    python -m scripts.run_lightgbm --trials 50
    python -m scripts.run_lightgbm --no-tune          # use default params
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.load import load_wide
from src.evaluation.backtest import aggregate_summaries, run_backtest
from src.models.lightgbm_model import DEFAULT_PARAMS, LGBMForecaster, tune_lightgbm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None, help="Optuna trials.")
    parser.add_argument("--no-tune", action="store_true", help="Skip tuning.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Wall-clock tuning limit in minutes (overrides the config value).",
    )
    args = parser.parse_args()

    cfg = load_config()
    horizon = int(cfg.get("data.horizon"))
    n_windows = int(cfg.get("data.n_backtest_windows"))
    n_trials = args.trials if args.trials is not None else int(cfg.get("models.optuna.n_trials"))

    print("Loading data ...")
    sales_wide, calendar, prices = load_wide(cfg)
    print(f"Series: {sales_wide['id'].nunique():,} | horizon: {horizon} | windows: {n_windows}")

    out_dir = cfg.path("paths.backtests_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = cfg.path("paths.models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.no_tune:
        params = dict(DEFAULT_PARAMS)
        print("Using default LightGBM parameters (tuning skipped).")
    else:
        print(f"Tuning LightGBM with Optuna ({n_trials} trials) ...", flush=True)
        params, best_value = tune_lightgbm(
            sales_wide,
            calendar,
            prices,
            cfg,
            n_trials=n_trials,
            horizon=horizon,
            timeout_minutes=args.timeout,
        )
        print(f"Best tuning WRMSSE: {best_value:.4f}")
        (models_dir / "lightgbm_best_params.json").write_text(json.dumps(params, indent=2))

    print("Backtesting tuned LightGBM ...")
    forecaster = LGBMForecaster(calendar, prices, cfg, params=params)
    summary, results = run_backtest(
        forecaster, sales_wide, calendar, prices, horizon, n_windows
    )
    summary.to_csv(out_dir / "lightgbm_windows.csv", index=False)

    forecaster.feature_importance().to_csv(
        out_dir / "lightgbm_feature_importance.csv", index=False
    )

    lgbm_agg = aggregate_summaries([summary])
    print("\nLightGBM backtest (mean over windows):\n")
    print(lgbm_agg.round(4).to_string(index=False))

    baseline_path = out_dir / "baselines_summary.csv"
    if baseline_path.exists():
        combined = pd.concat(
            [pd.read_csv(baseline_path), lgbm_agg], ignore_index=True
        ).sort_values("wrmsse_mean")
        combined.to_csv(out_dir / "leaderboard.csv", index=False)
        print("\nCombined leaderboard (lower WRMSSE is better):\n")
        print(combined.round(4).to_string(index=False))

    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
