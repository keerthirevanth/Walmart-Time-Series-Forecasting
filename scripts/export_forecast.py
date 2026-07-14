"""Fit the tuned LightGBM on all history and export the operational forecast.

Produces two artefacts consumed by the insight layer:

- forecast.parquet:        the 28-day-ahead point forecast per series.
- series_context.parquet:  a compact per-series summary (recent demand, zero
                           rate, price, category/store, forecast totals) so the
                           analyst can explain individual series without loading
                           the full history.

Uses the tuned hyperparameters from models/lightgbm_best_params.json when present,
otherwise the defaults.

Usage:
    python -m scripts.export_forecast
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.load import load_wide
from src.evaluation.metrics import ID_COLUMNS
from src.models.lightgbm_model import DEFAULT_PARAMS, LGBMForecaster


def _recent_stats(sales_wide: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    last_28 = sales_wide[day_cols[-28:]].to_numpy(dtype=float)
    last_7 = sales_wide[day_cols[-7:]].to_numpy(dtype=float)
    full = sales_wide[day_cols].to_numpy(dtype=float)
    stats = sales_wide[ID_COLUMNS].copy().reset_index(drop=True)
    stats["recent_mean_28"] = last_28.mean(axis=1).round(3)
    stats["recent_mean_7"] = last_7.mean(axis=1).round(3)
    stats["zero_rate"] = (full == 0).mean(axis=1).round(3)
    # A crude recent trend: last week versus the prior four-week average.
    stats["trend_7_vs_28"] = (stats["recent_mean_7"] - stats["recent_mean_28"]).round(3)
    return stats


def main() -> None:
    cfg = load_config()
    horizon = int(cfg.get("data.horizon"))

    print("Loading data ...", flush=True)
    sales_wide, calendar, prices = load_wide(cfg)
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    params_path = cfg.path("paths.models_dir") / "lightgbm_best_params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        print("Using tuned hyperparameters from", params_path)
    else:
        params = dict(DEFAULT_PARAMS)
        print("Tuned params not found; using defaults.")

    print(f"Fitting LightGBM on {sales_wide['id'].nunique():,} series and forecasting "
          f"{horizon} days ...", flush=True)
    forecaster = LGBMForecaster(calendar, prices, cfg, params=params)
    forecast_wide, info = forecaster.fit_predict(sales_wide, day_cols, horizon)
    print("Fit complete:", info, flush=True)

    forecast_cols = [c for c in forecast_wide.columns if c not in ID_COLUMNS]
    forecast_wide = forecast_wide.rename(
        columns={c: f"F{i + 1}" for i, c in enumerate(forecast_cols)}
    )

    # Per-series context for explanations.
    stats = _recent_stats(sales_wide, day_cols)
    fc_totals = forecast_wide.copy()
    f_cols = [c for c in fc_totals.columns if c.startswith("F")]
    fc_totals["forecast_total"] = fc_totals[f_cols].sum(axis=1).round(2)
    fc_totals["forecast_mean_28"] = fc_totals[f_cols].mean(axis=1).round(3)
    series_context = stats.merge(
        fc_totals[["id", "forecast_total", "forecast_mean_28"]], on="id", how="left"
    )

    out_dir = cfg.path("paths.forecast_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_wide.to_parquet(out_dir / "forecast.parquet", index=False)
    series_context.to_parquet(out_dir / "series_context.parquet", index=False)

    # Refresh the feature-importance artefact from this final fit.
    forecaster.feature_importance().to_csv(
        cfg.path("paths.backtests_dir") / "lightgbm_feature_importance.csv", index=False
    )

    print(f"Wrote forecast for {len(forecast_wide):,} series to {out_dir}")
    print(f"Total forecast units (next {horizon} days): "
          f"{series_context['forecast_total'].sum():,.0f}")


if __name__ == "__main__":
    main()
