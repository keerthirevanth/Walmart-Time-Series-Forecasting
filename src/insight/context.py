"""Load backtest and forecast artifacts and turn them into grounded context.

The analyst never invents numbers: every figure it can cite is assembled here
from the CSV/Parquet artifacts the pipeline wrote, and injected verbatim into the
prompt. If an artifact is missing the corresponding section is omitted rather
than fabricated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config


@dataclass
class InsightContext:
    leaderboard: pd.DataFrame | None
    feature_importance: pd.DataFrame | None
    best_params: dict | None
    series_context: pd.DataFrame | None

    # -- grounded text blocks --------------------------------------------
    def render_overview(self) -> str:
        blocks: list[str] = []

        if self.leaderboard is not None:
            blocks.append(
                "MODEL LEADERBOARD (rolling-origin backtest, lower WRMSSE is better):\n"
                + self.leaderboard.to_string(index=False)
            )

        if self.best_params is not None:
            keep = {
                k: self.best_params[k]
                for k in (
                    "objective",
                    "tweedie_variance_power",
                    "learning_rate",
                    "num_leaves",
                    "min_child_samples",
                )
                if k in self.best_params
            }
            blocks.append("TUNED LIGHTGBM HYPERPARAMETERS:\n" + json.dumps(keep, indent=2))

        if self.feature_importance is not None:
            top = self.feature_importance.head(12)
            blocks.append(
                "TOP LIGHTGBM FEATURES BY GAIN:\n" + top.to_string(index=False)
            )

        if self.series_context is not None:
            blocks.append("FORECAST AGGREGATES:\n" + self._aggregate_block())
            blocks.append(
                "CATEGORY RISK PROFILE (zero_rate is the share of zero-sales days; "
                "higher means more intermittent and harder to forecast, i.e. more "
                "risk; forecast_total is 28-day forecast volume):\n"
                + self.risk_profile()
            )

        if not blocks:
            return "No artifacts were found. Run the modelling and export steps first."
        return "\n\n".join(blocks)

    def _aggregate_block(self) -> str:
        sc = self.series_context
        lines = [
            f"Series covered: {len(sc):,}",
            f"Total forecast units (next 28 days): {sc['forecast_total'].sum():,.0f}",
        ]
        if "cat_id" in sc.columns:
            by_cat = sc.groupby("cat_id", observed=True)["forecast_total"].sum().sort_values(
                ascending=False
            )
            lines.append("By category:\n" + by_cat.to_string())
        if "store_id" in sc.columns:
            by_store = sc.groupby("store_id", observed=True)["forecast_total"].sum().sort_values(
                ascending=False
            )
            lines.append("By store:\n" + by_store.to_string())
        return "\n".join(lines)

    def risk_profile(self) -> str:
        """Per-category risk signals derived from the forecast context.

        There is no per-category WRMSSE artifact, so forecast risk is proxied by
        intermittency (zero rate) weighted by forecast volume: high-volume,
        highly intermittent categories are the hardest and costliest to get wrong.
        """
        if self.series_context is None:
            return "No per-series forecast context available."
        sc = self.series_context
        grp = sc.groupby("cat_id", observed=True).agg(
            n_series=("id", "count"),
            forecast_total=("forecast_total", "sum"),
            mean_zero_rate=("zero_rate", "mean"),
            intermittent_share=("zero_rate", lambda s: float((s > 0.6).mean())),
            mean_recent_demand=("recent_mean_28", "mean"),
        )
        grp = grp.round(3).sort_values("forecast_total", ascending=False)
        return grp.to_string()

    def top_series(self, n: int = 10, by: str = "forecast_total") -> str:
        if self.series_context is None:
            return "No per-series forecast context available."
        cols = [
            c
            for c in ["id", "cat_id", "store_id", "recent_mean_28", "forecast_total"]
            if c in self.series_context.columns
        ]
        top = self.series_context.sort_values(by, ascending=False).head(n)[cols]
        return top.to_string(index=False)

    def lookup_series(self, series_id: str) -> str:
        if self.series_context is None:
            return "No per-series forecast context available."
        row = self.series_context[self.series_context["id"] == series_id]
        if row.empty:
            # Allow a partial match on item id for convenience.
            row = self.series_context[
                self.series_context["id"].str.contains(series_id, regex=False)
            ]
        if row.empty:
            return f"No series matching '{series_id}' was found."
        return row.head(5).to_string(index=False)


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_context(cfg: Config) -> InsightContext:
    backtests = cfg.path("paths.backtests_dir")
    models_dir = cfg.path("paths.models_dir")
    forecast_dir = cfg.path("paths.forecast_dir")

    params_path = models_dir / "lightgbm_best_params.json"
    best_params = (
        json.loads(params_path.read_text()) if params_path.exists() else None
    )

    series_path = forecast_dir / "series_context.parquet"
    series_context = pd.read_parquet(series_path) if series_path.exists() else None

    return InsightContext(
        leaderboard=_read_csv(backtests / "leaderboard.csv"),
        feature_importance=_read_csv(backtests / "lightgbm_feature_importance.csv"),
        best_params=best_params,
        series_context=series_context,
    )
