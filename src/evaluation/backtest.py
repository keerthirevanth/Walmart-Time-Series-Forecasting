"""Rolling-origin backtesting.

Every model in the project is scored the same way: the history is cut at several
successive origins, each time forecasting the next `horizon` days and scoring
them with WRMSSE (and the secondary metrics). Averaging over multiple origins
gives a more stable read on accuracy than a single hold-out and mimics how a
demand forecast is actually re-run every period in production.

A "forecaster" here is any callable

    forecaster(train_wide, train_cols, horizon) -> forecast_wide

where `train_wide` is the id columns plus the training `d_*` columns available at
that origin, and the returned frame has the id columns plus `horizon` forecast
columns named to match the validation days.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    ID_COLUMNS,
    WRMSSEEvaluator,
    level_summary,
    mase,
    rmse,
)


class Forecaster(Protocol):
    name: str

    def __call__(
        self, train_wide: pd.DataFrame, train_cols: list[str], horizon: int
    ) -> pd.DataFrame:  # pragma: no cover - structural type
        ...


@dataclass
class WindowResult:
    window: int
    wrmsse: float
    mase: float
    rmse: float
    breakdown: pd.DataFrame


def rolling_origin_splits(
    day_cols: list[str], horizon: int, n_windows: int
) -> list[tuple[list[str], list[str]]]:
    """Return (train_cols, valid_cols) pairs from oldest origin to newest.

    The most recent `horizon` days form the last validation window, the block
    before it the previous one, and so on for `n_windows` windows.
    """
    n = len(day_cols)
    if n < horizon * (n_windows + 1):
        raise ValueError(
            f"Need at least {horizon * (n_windows + 1)} days for {n_windows} windows "
            f"of horizon {horizon}, but only {n} are available."
        )
    splits = []
    for k in range(n_windows, 0, -1):
        cut = n - k * horizon
        train_cols = day_cols[:cut]
        valid_cols = day_cols[cut : cut + horizon]
        splits.append((train_cols, valid_cols))
    return splits


def _point_metrics(
    train_wide: pd.DataFrame,
    valid_wide: pd.DataFrame,
    forecast_wide: pd.DataFrame,
    train_cols: list[str],
    valid_cols: list[str],
) -> tuple[float, float]:
    """Series-averaged MASE and RMSE at the bottom level of the hierarchy."""
    a = valid_wide[valid_cols].to_numpy(dtype=float)
    f = forecast_wide[valid_cols].to_numpy(dtype=float)
    train = train_wide[train_cols].to_numpy(dtype=float)

    rmse_val = rmse(a.ravel(), f.ravel())

    mase_vals = []
    for i in range(a.shape[0]):
        m = mase(a[i], f[i], train[i], seasonality=7)
        if not np.isnan(m):
            mase_vals.append(m)
    mase_val = float(np.mean(mase_vals)) if mase_vals else np.nan
    return mase_val, rmse_val


def run_backtest(
    forecaster: Forecaster,
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    n_windows: int,
) -> tuple[pd.DataFrame, list[WindowResult]]:
    """Run one forecaster across all rolling windows.

    Returns a one-row-per-window summary frame and the detailed per-window
    results (including the WRMSSE level breakdown for the last window).
    """
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    splits = rolling_origin_splits(day_cols, horizon, n_windows)

    results: list[WindowResult] = []
    for w, (train_cols, valid_cols) in enumerate(splits, start=1):
        window_start = time.time()
        print(
            f"  [{forecaster.name}] window {w}/{n_windows}: fitting and forecasting ...",
            flush=True,
        )
        train_wide = sales_wide[ID_COLUMNS + train_cols].copy()
        valid_wide = sales_wide[ID_COLUMNS + valid_cols].copy()

        forecast_wide = forecaster(train_wide, train_cols, horizon)
        forecast_wide = _align_forecast(forecast_wide, valid_wide, valid_cols)

        evaluator = WRMSSEEvaluator(train_wide, valid_wide, calendar, prices)
        wrmsse_val, breakdown = evaluator.score(forecast_wide)
        mase_val, rmse_val = _point_metrics(
            train_wide, valid_wide, forecast_wide, train_cols, valid_cols
        )
        print(
            f"  [{forecaster.name}] window {w}/{n_windows}: WRMSSE={wrmsse_val:.4f} "
            f"({time.time() - window_start:.0f}s)",
            flush=True,
        )

        results.append(
            WindowResult(
                window=w,
                wrmsse=wrmsse_val,
                mase=mase_val,
                rmse=rmse_val,
                breakdown=breakdown,
            )
        )

    summary = pd.DataFrame(
        {
            "model": forecaster.name,
            "window": [r.window for r in results],
            "wrmsse": [r.wrmsse for r in results],
            "mase": [r.mase for r in results],
            "rmse": [r.rmse for r in results],
        }
    )
    return summary, results


def _align_forecast(
    forecast_wide: pd.DataFrame, valid_wide: pd.DataFrame, valid_cols: list[str]
) -> pd.DataFrame:
    """Ensure the forecast has the id columns and the expected valid-day columns,
    ordered to match the actuals row for row."""
    fc = forecast_wide.copy()
    # Rename trailing columns to the validation day names if the forecaster used
    # generic horizon names.
    non_id = [c for c in fc.columns if c not in ID_COLUMNS]
    if non_id != valid_cols:
        if len(non_id) != len(valid_cols):
            raise ValueError(
                f"Forecast has {len(non_id)} horizon columns, expected {len(valid_cols)}."
            )
        fc = fc.rename(columns=dict(zip(non_id, valid_cols)))
    fc = fc[ID_COLUMNS + valid_cols]
    fc = fc.set_index("id").reindex(valid_wide["id"]).reset_index()
    # Restore the remaining id columns from the actuals (they align on id).
    for col in ID_COLUMNS:
        if col != "id":
            fc[col] = valid_wide[col].to_numpy()
    return fc[ID_COLUMNS + valid_cols]


def aggregate_summaries(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    """Average per-window metrics into one row per model for the results table."""
    combined = pd.concat(summaries, ignore_index=True)
    agg = (
        combined.groupby("model")
        .agg(
            wrmsse_mean=("wrmsse", "mean"),
            wrmsse_std=("wrmsse", "std"),
            mase_mean=("mase", "mean"),
            rmse_mean=("rmse", "mean"),
            n_windows=("window", "count"),
        )
        .reset_index()
        .sort_values("wrmsse_mean")
    )
    return agg
