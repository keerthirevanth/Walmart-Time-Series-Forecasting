"""Run the statistical baselines through the rolling-origin backtest.

Writes per-window and aggregated results to reports/backtests/ and prints a
markdown table suitable for pasting into notes or the README.

Usage:
    python -m scripts.run_baselines
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.load import load_wide
from src.evaluation.backtest import aggregate_summaries, run_backtest
from src.evaluation.metrics import level_summary
from src.models.baselines import default_baselines


def main() -> None:
    cfg = load_config()
    horizon = int(cfg.get("data.horizon"))
    n_windows = int(cfg.get("data.n_backtest_windows"))

    print("Loading data ...")
    sales_wide, calendar, prices = load_wide(cfg)
    n_series = sales_wide["id"].nunique()
    print(f"Series: {n_series:,} | horizon: {horizon} | windows: {n_windows}")

    summaries = []
    last_breakdowns = {}
    for forecaster in default_baselines():
        print(f"Backtesting baseline: {forecaster.name} ...")
        summary, results = run_backtest(
            forecaster, sales_wide, calendar, prices, horizon, n_windows
        )
        summaries.append(summary)
        last_breakdowns[forecaster.name] = results[-1].breakdown

    per_window = pd.concat(summaries, ignore_index=True)
    aggregate = aggregate_summaries(summaries)

    out_dir = cfg.path("paths.backtests_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_window.to_csv(out_dir / "baselines_windows.csv", index=False)
    aggregate.to_csv(out_dir / "baselines_summary.csv", index=False)

    # Per-level breakdown for the best baseline on the most recent window.
    best = aggregate.iloc[0]["model"]
    level_summary(last_breakdowns[best]).to_csv(
        out_dir / f"baselines_levels_{best}.csv", index=False
    )

    print("\nAggregated results (mean over windows, lower is better):\n")
    print(_to_markdown(aggregate))
    print(f"\nResults written to {out_dir}")


def _to_markdown(df: pd.DataFrame) -> str:
    cols = ["model", "wrmsse_mean", "wrmsse_std", "mase_mean", "rmse_mean", "n_windows"]
    df = df[cols].copy()
    for c in ["wrmsse_mean", "wrmsse_std", "mase_mean", "rmse_mean"]:
        df[c] = df[c].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
