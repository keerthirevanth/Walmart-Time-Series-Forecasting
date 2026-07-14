"""Backtest the Chronos foundation model zero-shot and add it to the leaderboard.

Chronos produces full quantile forecasts, so on top of the shared point metrics
(WRMSSE, MASE, RMSE) this script also reports the mean pinball loss, which scores
the calibration of the prediction intervals. Predictions are generated once per
window and reused for every metric.

Usage:
    python -m scripts.run_chronos
    python -m scripts.run_chronos --model amazon/chronos-bolt-small --batch-size 512
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.load import load_wide
from src.evaluation.backtest import _align_forecast, _point_metrics, rolling_origin_splits
from src.evaluation.metrics import ID_COLUMNS, WRMSSEEvaluator, mean_pinball_loss
from src.models.chronos_model import ChronosForecaster, _horizon_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Override the Chronos model name.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--point-estimate",
        choices=["mean", "median"],
        default="mean",
        help="Distribution summary used as the point forecast for WRMSSE/MASE/RMSE.",
    )
    args = parser.parse_args()

    cfg = load_config()
    horizon = int(cfg.get("data.horizon"))
    n_windows = int(cfg.get("data.n_backtest_windows"))

    print("Loading data ...")
    sales_wide, calendar, prices = load_wide(cfg)
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    splits = rolling_origin_splits(day_cols, horizon, n_windows)

    forecaster = ChronosForecaster(
        cfg,
        model_name=args.model,
        batch_size=args.batch_size,
        point_estimate=args.point_estimate,
    )
    forecaster.name = f"chronos_zeroshot_{args.point_estimate}"
    print(
        f"Model: {forecaster.model_name} | device: {forecaster.device} | "
        f"point estimate: {args.point_estimate}"
    )
    quantile_levels = forecaster.quantile_levels

    rows = []
    for w, (train_cols, valid_cols) in enumerate(splits, start=1):
        print(f"Window {w}/{n_windows}: forecasting {len(sales_wide):,} series ...")
        train_wide = sales_wide[ID_COLUMNS + train_cols].copy()
        valid_wide = sales_wide[ID_COLUMNS + valid_cols].copy()

        median, quantiles = forecaster.predict_quantiles(train_wide, train_cols, horizon)

        forecast_wide = train_wide[ID_COLUMNS].copy().reset_index(drop=True)
        forecast_wide[_horizon_columns(horizon)] = median
        forecast_wide = _align_forecast(forecast_wide, valid_wide, valid_cols)

        evaluator = WRMSSEEvaluator(train_wide, valid_wide, calendar, prices)
        wrmsse_val, _ = evaluator.score(forecast_wide)
        mase_val, rmse_val = _point_metrics(
            train_wide, valid_wide, forecast_wide, train_cols, valid_cols
        )

        actual = valid_wide[valid_cols].to_numpy(dtype=float).ravel()
        qf = {
            q: quantiles[:, :, i].ravel() for i, q in enumerate(quantile_levels)
        }
        pinball = mean_pinball_loss(actual, qf)

        rows.append(
            {
                "model": forecaster.name,
                "window": w,
                "wrmsse": wrmsse_val,
                "mase": mase_val,
                "rmse": rmse_val,
                "pinball": pinball,
            }
        )
        print(f"  WRMSSE={wrmsse_val:.4f}  MASE={mase_val:.4f}  pinball={pinball:.4f}")

    summary = pd.DataFrame(rows)
    out_dir = cfg.path("paths.backtests_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.point_estimate
    summary.to_csv(out_dir / f"chronos_{tag}_windows.csv", index=False)

    agg = (
        summary.groupby("model")
        .agg(
            wrmsse_mean=("wrmsse", "mean"),
            wrmsse_std=("wrmsse", "std"),
            mase_mean=("mase", "mean"),
            rmse_mean=("rmse", "mean"),
            pinball_mean=("pinball", "mean"),
            n_windows=("window", "count"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / f"chronos_{tag}_summary.csv", index=False)

    print("\nChronos backtest (mean over windows):\n")
    print(agg.round(4).to_string(index=False))
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
