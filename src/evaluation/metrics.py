"""Forecast accuracy metrics used throughout the project.

The headline metric is WRMSSE, the official metric of the M5 Accuracy
competition. It is a weighted average of RMSSE across the 12 levels of the
product hierarchy, where the weight of each series is its share of dollar sales
over the last 28 days of the training window. Getting this metric right is the
single most important measurement decision in the project, so it is implemented
here from first principles rather than approximated.

Point metrics (MASE, RMSE) and the probabilistic pinball loss are also provided
so that models producing prediction intervals can be scored on their calibrated
uncertainty, not just their point accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simple point / probabilistic metrics
# ---------------------------------------------------------------------------
def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train_series: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    The scale is the mean absolute seasonal-naive error on the training series.
    ``seasonality=1`` gives the standard (non-seasonal) MASE; use 7 for weekly
    seasonality on daily data.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    train_series = np.asarray(train_series, dtype=float)

    if len(train_series) <= seasonality:
        return np.nan
    scale = np.mean(np.abs(train_series[seasonality:] - train_series[:-seasonality]))
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(actual - forecast)) / scale)


def pinball_loss(
    actual: np.ndarray, quantile_forecast: np.ndarray, quantile: float
) -> float:
    """Pinball (quantile) loss for a single quantile level."""
    actual = np.asarray(actual, dtype=float)
    quantile_forecast = np.asarray(quantile_forecast, dtype=float)
    delta = actual - quantile_forecast
    loss = np.maximum(quantile * delta, (quantile - 1) * delta)
    return float(np.mean(loss))


def mean_pinball_loss(
    actual: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> float:
    """Average pinball loss across a set of quantile forecasts.

    ``quantile_forecasts`` maps a quantile level (e.g. 0.5) to its forecast array.
    """
    losses = [
        pinball_loss(actual, forecast, q) for q, forecast in quantile_forecasts.items()
    ]
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# WRMSSE
# ---------------------------------------------------------------------------
# The 12 aggregation levels defined by the competition. An empty grouping list
# means the grand total across all series.
WRMSSE_LEVELS: list[list[str]] = [
    [],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


class WRMSSEEvaluator:
    """Compute WRMSSE for M5-style hierarchical forecasts.

    Parameters
    ----------
    train_df:
        Wide sales frame with the id columns plus one ``d_*`` column per training
        day. This is the history available to the model.
    valid_df:
        Wide sales frame with the id columns plus one ``d_*`` column per horizon
        day (the actuals to score against).
    calendar:
        The M5 calendar table (maps ``d_*`` to ``wm_yr_wk`` etc.).
    prices:
        The M5 sell_prices table used to compute dollar-sales weights.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        calendar: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> None:
        self.calendar = calendar
        self.prices = prices

        self.train_day_cols = [c for c in train_df.columns if c.startswith("d_")]
        self.valid_day_cols = [c for c in valid_df.columns if c.startswith("d_")]
        self.horizon = len(self.valid_day_cols)

        self.train_df = train_df.copy()
        self.valid_df = valid_df.copy()

        self.group_ids = self._build_group_index()
        self.weights = self._compute_weights()
        self.scales = self._compute_scales()

    # -- hierarchy roll-up -------------------------------------------------
    def _rollup(self, df: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
        """Stack the actuals/forecasts for all 12 levels into one long frame."""
        frames = []
        for level, keys in enumerate(WRMSSE_LEVELS):
            if keys:
                agg = df.groupby(keys, observed=True)[day_cols].sum()
                index = agg.index.to_frame(index=False).astype(str).agg("--".join, axis=1)
            else:
                agg = df[day_cols].sum().to_frame().T
                index = pd.Series(["Total"])
            agg = agg.reset_index(drop=True)
            agg.insert(0, "series_key", (str(level) + "::") + index.values)
            frames.append(agg)
        return pd.concat(frames, ignore_index=True)

    def _build_group_index(self) -> pd.Series:
        rolled = self._rollup(self.train_df, self.train_day_cols[:1])
        return rolled["series_key"]

    # -- weights -----------------------------------------------------------
    def _dollar_sales_last_28(self) -> pd.Series:
        """Dollar sales per bottom-level series over the last 28 training days."""
        last_cols = self.train_day_cols[-28:]
        long = self.train_df.melt(
            id_vars=ID_COLUMNS, value_vars=last_cols, var_name="d", value_name="sales"
        )
        long = long.merge(self.calendar[["d", "wm_yr_wk"]], on="d", how="left")
        long = long.merge(
            self.prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
        )
        long["dollar_sales"] = long["sales"] * long["sell_price"].fillna(0.0)
        return long.groupby("id", observed=True)["dollar_sales"].sum()

    def _compute_weights(self) -> pd.Series:
        dollars = self._dollar_sales_last_28()
        base = self.train_df[ID_COLUMNS].copy()
        base["dollar_sales"] = base["id"].map(dollars).fillna(0.0)

        weights = {}
        for level, keys in enumerate(WRMSSE_LEVELS):
            if keys:
                grouped = base.groupby(keys, observed=True)["dollar_sales"].sum()
                index = grouped.index.to_frame(index=False).astype(str).agg(
                    "--".join, axis=1
                )
                grouped.index = (str(level) + "::") + index.values
            else:
                grouped = pd.Series(
                    {f"{level}::Total": base["dollar_sales"].sum()}
                )
            total = grouped.sum()
            # Each level contributes equally (1/12), weighted within level by
            # dollar-sales share.
            weights.update((grouped / total / len(WRMSSE_LEVELS)).to_dict())
        return pd.Series(weights)

    # -- scales (RMSSE denominator) ---------------------------------------
    def _compute_scales(self) -> pd.Series:
        rolled = self._rollup(self.train_df, self.train_day_cols)
        values = rolled.drop(columns=["series_key"]).to_numpy(dtype=float)
        keys = rolled["series_key"].to_numpy()

        scales = np.empty(len(values))
        for i, series in enumerate(values):
            # Ignore the leading zeros before the first observed sale: those days
            # predate the product's introduction and would deflate the scale.
            nz = np.flatnonzero(series)
            if nz.size == 0:
                scales[i] = np.nan
                continue
            active = series[nz[0]:]
            diffs = np.diff(active)
            scales[i] = np.mean(diffs**2) if diffs.size else np.nan
        return pd.Series(scales, index=keys)

    # -- scoring -----------------------------------------------------------
    def score(self, forecast_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Score a wide forecast frame (id columns + horizon ``d_*`` columns).

        Returns the scalar WRMSSE and a per-level breakdown so we can see which
        parts of the hierarchy a model gets right or wrong.
        """
        forecast_df = forecast_df.copy()
        # Accept forecasts whose horizon columns use different names (e.g. h_1..h_n)
        # as long as the id columns lead and the column count matches.
        if list(forecast_df.columns[: len(ID_COLUMNS)]) == ID_COLUMNS:
            expected = ID_COLUMNS + self.valid_day_cols[: self.horizon]
            if len(forecast_df.columns) != len(expected):
                raise ValueError(
                    f"Forecast has {len(forecast_df.columns)} columns, expected "
                    f"{len(expected)} (id columns plus {self.horizon} horizon days)."
                )
            forecast_df.columns = expected

        actual_rolled = self._rollup(self.valid_df, self.valid_day_cols)
        fcst_rolled = self._rollup(forecast_df, self.valid_day_cols)

        actual_rolled = actual_rolled.set_index("series_key")
        fcst_rolled = fcst_rolled.set_index("series_key").reindex(actual_rolled.index)

        a = actual_rolled.to_numpy(dtype=float)
        f = fcst_rolled.to_numpy(dtype=float)
        keys = actual_rolled.index

        mse = np.mean((a - f) ** 2, axis=1)
        scale = self.scales.reindex(keys).to_numpy()
        rmsse = np.sqrt(mse / scale)

        weight = self.weights.reindex(keys).to_numpy()
        contribution = weight * rmsse

        breakdown = pd.DataFrame(
            {
                "series_key": keys,
                "level": [int(k.split("::", 1)[0]) for k in keys],
                "rmsse": rmsse,
                "weight": weight,
                "contribution": contribution,
            }
        )
        valid = breakdown.dropna(subset=["contribution"])
        wrmsse = float(valid["contribution"].sum())
        return wrmsse, breakdown


def level_summary(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a WRMSSE breakdown to per-level contributions."""
    return (
        breakdown.dropna(subset=["contribution"])
        .groupby("level")
        .agg(
            series=("series_key", "count"),
            mean_rmsse=("rmsse", "mean"),
            contribution=("contribution", "sum"),
        )
        .reset_index()
    )
