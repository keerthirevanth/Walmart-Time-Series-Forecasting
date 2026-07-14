"""Tests for the LLM insight layer, using the offline echo provider.

The echo provider returns the assembled prompt, so we can assert that the
grounded numbers actually reach the model and that the three analyst capabilities
run end to end without any API key or network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.insight.analyst import ForecastAnalyst, SYSTEM_PROMPT
from src.insight.context import InsightContext
from src.insight.llm_client import EchoProvider, RotatingProvider, get_provider


class _FailingProvider:
    name = "failing"

    def complete(self, system, prompt):
        raise RuntimeError("rate limit")


class _OkProvider:
    name = "ok"

    def __init__(self, text="ok-response"):
        self.text = text

    def complete(self, system, prompt):
        return self.text


def _context() -> InsightContext:
    leaderboard = pd.DataFrame(
        {
            "model": ["lightgbm", "seasonal_naive", "chronos_zeroshot_mean"],
            "wrmsse_mean": [0.602, 0.752, 1.924],
            "mase_mean": [1.815, 2.013, 1.528],
        }
    )
    feature_importance = pd.DataFrame(
        {"feature": ["rmean_28", "item_id", "sell_price"], "gain": [2.8e6, 4.8e5, 2.3e5]}
    )
    series_context = pd.DataFrame(
        {
            "id": ["FOODS_3_090_CA_1_evaluation", "HOBBIES_1_001_CA_2_evaluation"],
            "item_id": ["FOODS_3_090", "HOBBIES_1_001"],
            "cat_id": ["FOODS", "HOBBIES"],
            "store_id": ["CA_1", "CA_2"],
            "recent_mean_28": [12.5, 0.3],
            "recent_mean_7": [14.0, 0.1],
            "zero_rate": [0.05, 0.82],
            "trend_7_vs_28": [1.5, -0.2],
            "forecast_total": [360.0, 6.0],
            "forecast_mean_28": [12.9, 0.21],
        }
    )
    return InsightContext(
        leaderboard=leaderboard,
        feature_importance=feature_importance,
        best_params={"objective": "tweedie", "tweedie_variance_power": 1.06},
        series_context=series_context,
    )


def _analyst() -> ForecastAnalyst:
    return ForecastAnalyst(load_config(), EchoProvider(), _context())


def test_get_provider_echo_needs_no_key():
    prov = get_provider(load_config(), override="echo")
    assert prov.name == "echo"


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError):
        get_provider(load_config(), override="does-not-exist")


def test_rotation_falls_back_past_failure():
    rot = RotatingProvider([_FailingProvider(), _OkProvider("second-wins")])
    assert rot.complete("sys", "prompt") == "second-wins"
    assert "ok" in rot.name  # name reflects the provider that served the call


def test_rotation_all_fail_raises():
    rot = RotatingProvider([_FailingProvider(), _FailingProvider()])
    with pytest.raises(RuntimeError):
        rot.complete("sys", "prompt")


def test_rotation_empty_raises():
    with pytest.raises(ValueError):
        RotatingProvider([])


def test_overview_contains_grounded_numbers():
    overview = _context().render_overview()
    assert "0.602" in overview          # leaderboard WRMSSE
    assert "rmean_28" in overview        # top feature
    assert "tweedie_variance_power" in overview


def test_executive_summary_grounds_on_context():
    out = _analyst().executive_summary()
    assert SYSTEM_PROMPT[:20] in out     # echo includes the system prompt
    assert "0.602" in out                # the real number reached the prompt


def test_answer_includes_top_series():
    out = _analyst().answer("Which series has the highest forecast volume?")
    assert "FOODS_3_090_CA_1_evaluation" in out
    assert "highest forecast volume" in out.lower()


def test_explain_series_finds_by_substring():
    out = _analyst().explain_series("HOBBIES_1_001")
    # The intermittent series' facts must reach the prompt.
    assert "0.82" in out                 # its zero rate
    assert "HOBBIES" in out


def test_explain_missing_series_reports_cleanly():
    out = _analyst().explain_series("NOT_A_REAL_ID")
    assert "No series matching" in out
