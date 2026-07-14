"""Tests for the Chronos forecaster's own logic, with the model mocked out.

torch and chronos-forecasting are heavy GPU dependencies, so they are not
installed in the test environment. These tests inject a fake pipeline and a fake
torch to verify the parts that are this project's responsibility: context
selection, batching, quantile indexing, clipping, and the shape/columns of the
assembled forecast. The model's numerical output is not under test here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import ID_COLUMNS
from src.models.chronos_model import ChronosForecaster, _horizon_columns


class _FakeQuantileTensor:
    def __init__(self, array):
        self._array = array

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeTorch:
    def tensor(self, row):
        return np.asarray(row)


class _FakePipeline:
    """Returns quantiles that deliberately include negatives so clipping is tested."""

    def predict_quantiles(self, context, prediction_length, quantile_levels):
        batch = len(context)
        n_levels = len(quantile_levels)
        # Shape (batch, horizon, n_levels); fill each level with its index minus one
        # so some values are negative (level 0 -> -1) and clipping must fire.
        q = np.zeros((batch, prediction_length, n_levels), dtype=np.float32)
        for i in range(n_levels):
            q[:, :, i] = i - 1
        mean = q.mean(axis=2)
        return _FakeQuantileTensor(q), _FakeQuantileTensor(mean)


def _make_forecaster(batch_size=2, point_estimate="median"):
    cfg = load_config()
    fc = object.__new__(ChronosForecaster)
    fc.cfg = cfg
    fc.model_name = "fake"
    fc.device = "cpu"
    fc.batch_size = batch_size
    fc.quantile_levels = list(cfg.get("evaluation.quantiles"))
    fc.point_estimate = point_estimate
    fc.name = "chronos_zeroshot"
    fc._pipeline = _FakePipeline()
    fc._torch = _FakeTorch()
    return fc


def _panel(n_series=5, n_days=40):
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    rng = np.random.default_rng(0)
    records = []
    for s in range(n_series):
        base = {
            "id": f"S{s}",
            "item_id": f"I{s}",
            "dept_id": "D",
            "cat_id": "C",
            "store_id": "ST",
            "state_id": "CA",
        }
        records.append({**base, **{c: rng.integers(0, 5) for c in day_cols}})
    return pd.DataFrame(records), day_cols


def test_predict_quantiles_shapes_and_clipping():
    fc = _make_forecaster(batch_size=2)
    panel, day_cols = _panel(n_series=5)
    horizon = 7
    median, quantiles = fc.predict_quantiles(panel, day_cols, horizon)

    n_levels = len(fc.quantile_levels)
    assert median.shape == (5, horizon)
    assert quantiles.shape == (5, horizon, n_levels)
    # The fake returns negative values at the lowest level; output must be clipped.
    assert (quantiles >= 0).all()
    assert (median >= 0).all()


def test_median_uses_the_0_5_level():
    fc = _make_forecaster()
    panel, day_cols = _panel(n_series=3)
    _, quantiles = fc.predict_quantiles(panel, day_cols, horizon=4)
    median, _ = fc.predict_quantiles(panel, day_cols, horizon=4)
    mid = fc.quantile_levels.index(0.5)
    np.testing.assert_allclose(median, quantiles[:, :, mid])


def test_mean_point_estimate_uses_returned_mean():
    # With 9 quantile levels valued [-1, 0, ..., 7] in the fake, the returned
    # mean is their average = 3.0; the point forecast should equal it (clipped).
    fc = _make_forecaster(point_estimate="mean")
    panel, day_cols = _panel(n_series=3)
    point, _ = fc.predict_quantiles(panel, day_cols, horizon=4)
    np.testing.assert_allclose(point, 3.0)


def test_call_returns_wide_with_id_and_horizon_columns():
    fc = _make_forecaster()
    panel, day_cols = _panel(n_series=4)
    horizon = 5
    out = fc(panel, day_cols, horizon)
    assert list(out["id"]) == list(panel["id"])
    for col in ID_COLUMNS:
        assert col in out.columns
    for col in _horizon_columns(horizon):
        assert col in out.columns
    assert len(out) == 4


def test_batching_matches_single_batch():
    panel, day_cols = _panel(n_series=6)
    small = _make_forecaster(batch_size=2)
    big = _make_forecaster(batch_size=100)
    m_small, _ = small.predict_quantiles(panel, day_cols, horizon=3)
    m_big, _ = big.predict_quantiles(panel, day_cols, horizon=3)
    np.testing.assert_allclose(m_small, m_big)
