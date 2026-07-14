"""Known-answer tests for the accuracy metrics.

These run without the real dataset: a tiny synthetic hierarchy is enough to pin
down the WRMSSE behaviour (perfect forecast scores zero, weights across the 12
levels sum to one) and to check the point/probabilistic metrics against hand
computable values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    WRMSSEEvaluator,
    mase,
    mean_pinball_loss,
    pinball_loss,
    rmse,
)


# ---------------------------------------------------------------------------
# Point / probabilistic metrics
# ---------------------------------------------------------------------------
def test_rmse_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    actual = np.array([0.0, 0.0, 0.0])
    forecast = np.array([1.0, 1.0, 1.0])
    assert rmse(actual, forecast) == pytest.approx(1.0)


def test_mase_scales_by_naive_error():
    train = np.array([1.0, 2.0, 1.0, 2.0, 1.0])  # naive abs error = 1 each step
    actual = np.array([3.0, 3.0])
    forecast = np.array([2.0, 2.0])  # abs error = 1 each step
    assert mase(actual, forecast, train, seasonality=1) == pytest.approx(1.0)


def test_pinball_median_equals_half_mae():
    actual = np.array([0.0, 2.0])
    forecast = np.array([1.0, 1.0])
    # At q=0.5 the pinball loss is half the absolute error.
    assert pinball_loss(actual, forecast, 0.5) == pytest.approx(0.5)


def test_mean_pinball_symmetric_quantiles():
    actual = np.array([10.0, 10.0])
    forecasts = {0.1: np.array([9.0, 9.0]), 0.9: np.array([11.0, 11.0])}
    # Both quantiles are off by 1 on the favourable side; average stays positive.
    assert mean_pinball_loss(actual, forecasts) > 0


# ---------------------------------------------------------------------------
# WRMSSE on a synthetic 4-series hierarchy
# ---------------------------------------------------------------------------
def _synthetic_m5(n_train: int = 30, horizon: int = 3):
    ids = []
    rows = []
    # 2 states, 1 store each, 2 items each -> 4 bottom-level series.
    layout = [
        ("CA", "CA_1", "ITEM_1"),
        ("CA", "CA_1", "ITEM_2"),
        ("TX", "TX_1", "ITEM_1"),
        ("TX", "TX_1", "ITEM_2"),
    ]
    rng = np.random.default_rng(0)
    train_cols = [f"d_{i}" for i in range(1, n_train + 1)]
    valid_cols = [f"d_{i}" for i in range(n_train + 1, n_train + horizon + 1)]

    train_records, valid_records = [], []
    for state, store, item in layout:
        series_id = f"{item}_{store}_evaluation"
        base = {
            "id": series_id,
            "item_id": item,
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "store_id": store,
            "state_id": state,
        }
        train_vals = rng.integers(0, 5, size=n_train)
        valid_vals = rng.integers(0, 5, size=horizon)
        train_records.append({**base, **dict(zip(train_cols, train_vals))})
        valid_records.append({**base, **dict(zip(valid_cols, valid_vals))})

    train_df = pd.DataFrame(train_records)
    valid_df = pd.DataFrame(valid_records)

    all_days = train_cols + valid_cols
    calendar = pd.DataFrame(
        {"d": all_days, "wm_yr_wk": [11101 + (i // 7) for i in range(len(all_days))]}
    )
    # One price per (store, item, week) covering all weeks present.
    price_rows = []
    weeks = calendar["wm_yr_wk"].unique()
    for state, store, item in layout:
        for wk in weeks:
            price_rows.append(
                {"store_id": store, "item_id": item, "wm_yr_wk": wk, "sell_price": 2.0}
            )
    prices = pd.DataFrame(price_rows)
    return train_df, valid_df, calendar, prices, valid_cols


def test_wrmsse_weights_sum_to_one():
    train_df, valid_df, calendar, prices, _ = _synthetic_m5()
    ev = WRMSSEEvaluator(train_df, valid_df, calendar, prices)
    # 12 levels, each contributing weight 1/12 -> total 1.
    assert ev.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_wrmsse_zero_for_perfect_forecast():
    train_df, valid_df, calendar, prices, valid_cols = _synthetic_m5()
    ev = WRMSSEEvaluator(train_df, valid_df, calendar, prices)
    forecast = valid_df.copy()  # forecast equals the actuals
    score, breakdown = ev.score(forecast)
    assert score == pytest.approx(0.0, abs=1e-9)
    assert (breakdown["rmsse"].dropna() == 0).all()


def test_wrmsse_positive_for_biased_forecast():
    train_df, valid_df, calendar, prices, valid_cols = _synthetic_m5()
    ev = WRMSSEEvaluator(train_df, valid_df, calendar, prices)
    forecast = valid_df.copy()
    for col in valid_cols:
        forecast[col] = forecast[col] + 3  # constant upward bias
    score, _ = ev.score(forecast)
    assert score > 0
