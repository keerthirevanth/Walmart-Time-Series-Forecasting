"""Statistical baseline forecasters.

Every learned model is judged against these. They are deliberately cheap and
vectorised so they run over the full subset in seconds:

- naive           last observed value, held flat over the horizon.
- seasonal_naive  the last seven days, tiled across the horizon (weekly cycle).
- moving_average  the mean of the last N days, held flat.
- croston_sba     Croston's method with the Syntetos-Boylan approximation, the
                  standard baseline for intermittent demand.

Each forecaster returns the id columns plus `horizon` forecast columns named
`h_1 .. h_horizon`; the backtest harness renames them to the validation days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import ID_COLUMNS


def _horizon_columns(horizon: int) -> list[str]:
    return [f"h_{i}" for i in range(1, horizon + 1)]


def _assemble(train_wide: pd.DataFrame, forecast: np.ndarray, horizon: int) -> pd.DataFrame:
    cols = _horizon_columns(horizon)
    out = train_wide[ID_COLUMNS].copy().reset_index(drop=True)
    out[cols] = forecast
    return out


class NaiveForecaster:
    name = "naive"

    def __call__(self, train_wide, train_cols, horizon):
        last = train_wide[train_cols[-1]].to_numpy(dtype=float)
        forecast = np.repeat(last[:, None], horizon, axis=1)
        return _assemble(train_wide, forecast, horizon)


class SeasonalNaiveForecaster:
    name = "seasonal_naive"

    def __init__(self, season_length: int = 7):
        self.season_length = season_length

    def __call__(self, train_wide, train_cols, horizon):
        season = train_wide[train_cols[-self.season_length :]].to_numpy(dtype=float)
        # Tile the last season across the horizon.
        reps = int(np.ceil(horizon / self.season_length))
        tiled = np.tile(season, (1, reps))[:, :horizon]
        return _assemble(train_wide, tiled, horizon)


class MovingAverageForecaster:
    name = "moving_average"

    def __init__(self, window: int = 28):
        self.window = window

    def __call__(self, train_wide, train_cols, horizon):
        recent = train_wide[train_cols[-self.window :]].to_numpy(dtype=float)
        mean = recent.mean(axis=1, keepdims=True)
        forecast = np.repeat(mean, horizon, axis=1)
        return _assemble(train_wide, forecast, horizon)


def _croston_rate(series: np.ndarray, alpha: float, sba: bool) -> float:
    """Croston / SBA demand-rate estimate for a single series."""
    nz_idx = np.flatnonzero(series)
    if nz_idx.size == 0:
        return 0.0
    sizes = series[nz_idx].astype(float)
    # Inter-arrival intervals: first interval measured from the series start.
    intervals = np.diff(np.concatenate(([-1], nz_idx))).astype(float)

    z = sizes[0]
    p = intervals[0]
    for i in range(1, len(sizes)):
        z += alpha * (sizes[i] - z)
        p += alpha * (intervals[i] - p)
    rate = z / p if p > 0 else 0.0
    if sba:
        rate *= 1.0 - alpha / 2.0
    return rate


class CrostonSBAForecaster:
    name = "croston_sba"

    def __init__(self, alpha: float = 0.1, sba: bool = True):
        self.alpha = alpha
        self.sba = sba

    def __call__(self, train_wide, train_cols, horizon):
        values = train_wide[train_cols].to_numpy(dtype=float)
        rates = np.array(
            [_croston_rate(row, self.alpha, self.sba) for row in values]
        )
        forecast = np.repeat(rates[:, None], horizon, axis=1)
        return _assemble(train_wide, forecast, horizon)


def default_baselines() -> list:
    """The baseline set referenced from the config, ready to run."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(season_length=7),
        MovingAverageForecaster(window=28),
        CrostonSBAForecaster(alpha=0.1, sba=True),
    ]
