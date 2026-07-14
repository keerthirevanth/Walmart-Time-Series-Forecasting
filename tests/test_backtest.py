"""Tests for the rolling-origin backtest harness and the baselines.

Uses a small synthetic M5-shaped panel so the harness and forecasters can be
checked for shape, ordering, and sane scores without the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    aggregate_summaries,
    rolling_origin_splits,
    run_backtest,
)
from src.models.baselines import (
    CrostonSBAForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    default_baselines,
)


def _synthetic_panel(n_days: int = 40):
    layout = [
        ("CA", "CA_1", "ITEM_1"),
        ("CA", "CA_1", "ITEM_2"),
        ("TX", "TX_1", "ITEM_1"),
        ("TX", "TX_1", "ITEM_2"),
    ]
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    rng = np.random.default_rng(1)
    records = []
    for state, store, item in layout:
        base = {
            "id": f"{item}_{store}_evaluation",
            "item_id": item,
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "store_id": store,
            "state_id": state,
        }
        vals = rng.integers(0, 6, size=n_days)
        records.append({**base, **dict(zip(day_cols, vals))})
    sales_wide = pd.DataFrame(records)

    calendar = pd.DataFrame(
        {"d": day_cols, "wm_yr_wk": [11101 + (i // 7) for i in range(n_days)]}
    )
    weeks = calendar["wm_yr_wk"].unique()
    price_rows = [
        {"store_id": store, "item_id": item, "wm_yr_wk": wk, "sell_price": 2.0}
        for _, store, item in layout
        for wk in weeks
    ]
    prices = pd.DataFrame(price_rows)
    return sales_wide, calendar, prices


def test_rolling_origin_splits_shapes():
    day_cols = [f"d_{i}" for i in range(1, 31)]
    splits = rolling_origin_splits(day_cols, horizon=4, n_windows=3)
    assert len(splits) == 3
    # Oldest origin first, newest last.
    assert splits[0][0][-1] == "d_18"  # train end of first window
    assert splits[-1][1] == ["d_27", "d_28", "d_29", "d_30"]  # last valid window
    for train_cols, valid_cols in splits:
        assert len(valid_cols) == 4
        assert set(train_cols).isdisjoint(valid_cols)


def test_rolling_origin_requires_enough_history():
    day_cols = [f"d_{i}" for i in range(1, 10)]
    with pytest.raises(ValueError):
        rolling_origin_splits(day_cols, horizon=4, n_windows=3)


def test_baseline_forecast_shapes_and_order():
    sales_wide, _, _ = _synthetic_panel()
    train_cols = [c for c in sales_wide.columns if c.startswith("d_")][:30]
    fc = NaiveForecaster()(sales_wide, train_cols, horizon=5)
    assert list(fc["id"]) == list(sales_wide["id"])  # row order preserved
    assert fc.shape[0] == sales_wide.shape[0]
    horizon_cols = [c for c in fc.columns if c.startswith("h_")]
    assert len(horizon_cols) == 5


def test_seasonal_naive_repeats_last_week():
    sales_wide, _, _ = _synthetic_panel()
    train_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    fc = SeasonalNaiveForecaster(season_length=7)(sales_wide, train_cols, horizon=7)
    last_week = sales_wide[train_cols[-7:]].to_numpy(dtype=float)
    got = fc[[c for c in fc.columns if c.startswith("h_")]].to_numpy()
    np.testing.assert_allclose(got, last_week)


def test_croston_rate_matches_mean_rate_on_regular_series():
    # A perfectly regular series of constant demand should give a rate close to
    # that constant.
    fc = CrostonSBAForecaster(alpha=0.3, sba=False)
    series = np.array([2.0] * 20)
    sales_wide = pd.DataFrame(
        [
            {
                "id": "X",
                "item_id": "I",
                "dept_id": "D",
                "cat_id": "C",
                "store_id": "S",
                "state_id": "ST",
                **{f"d_{i+1}": series[i] for i in range(len(series))},
            }
        ]
    )
    train_cols = [f"d_{i+1}" for i in range(len(series))]
    out = fc(sales_wide, train_cols, horizon=3)
    assert out.iloc[0]["h_1"] == pytest.approx(2.0, abs=1e-6)


def test_run_backtest_end_to_end():
    sales_wide, calendar, prices = _synthetic_panel(n_days=40)
    summary, results = run_backtest(
        NaiveForecaster(), sales_wide, calendar, prices, horizon=4, n_windows=2
    )
    assert len(summary) == 2
    assert (summary["wrmsse"] >= 0).all()
    assert summary["wrmsse"].notna().all()
    assert len(results) == 2


def test_aggregate_summaries_orders_by_wrmsse():
    sales_wide, calendar, prices = _synthetic_panel(n_days=40)
    summaries = []
    for fc in default_baselines():
        s, _ = run_backtest(fc, sales_wide, calendar, prices, horizon=4, n_windows=2)
        summaries.append(s)
    agg = aggregate_summaries(summaries)
    assert list(agg["model"])  # non-empty
    assert agg["wrmsse_mean"].is_monotonic_increasing
