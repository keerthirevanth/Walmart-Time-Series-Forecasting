"""Global LightGBM forecaster with a Tweedie objective and Optuna tuning.

A single model is trained across all series (a "global" model), which lets it
borrow strength across the many sparse, intermittent series that dominate this
data. The Tweedie objective is chosen deliberately: with 66% zero-sales days the
target is a zero-inflated non-negative count, exactly what Tweedie regression is
built for. The Tweedie variance power is treated as a hyperparameter rather than
fixed by assumption.

The forecaster follows the same call signature as the baselines so it runs
through the shared rolling-origin backtest unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import Config
from src.features.build_features import build_features
from src.evaluation.metrics import ID_COLUMNS, WRMSSEEvaluator


@dataclass
class PreparedData:
    """Reusable training artefacts so hyperparameter search need not rebuild
    features for every trial."""

    train_set: Any
    valid_set: Any
    future: pd.DataFrame
    feature_cols: list[str]
    future_cols: list[str]


def _day_index(day_col: str) -> int:
    return int(day_col.split("_")[1])


def _future_day_cols(train_cols: list[str], horizon: int) -> list[str]:
    last = _day_index(train_cols[-1])
    return [f"d_{last + i}" for i in range(1, horizon + 1)]


def build_extended_long(
    train_wide: pd.DataFrame,
    train_cols: list[str],
    horizon: int,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Melt the training window to long and append placeholder future rows.

    The future rows (NaN sales) are what the model ultimately predicts; keeping
    them in the same frame means the lag/rolling features are computed once and
    stay horizon-safe.
    """
    future_cols = _future_day_cols(train_cols, horizon)

    long_hist = train_wide.melt(
        id_vars=ID_COLUMNS, value_vars=train_cols, var_name="d", value_name="sales"
    )

    ids = train_wide[ID_COLUMNS].drop_duplicates()
    future = ids.merge(pd.DataFrame({"d": future_cols}), how="cross")
    future["sales"] = np.nan

    long_all = pd.concat([long_hist, future], ignore_index=True)

    cal_cols = [
        "d", "date", "wm_yr_wk", "wday", "month", "year",
        "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
    ]
    long_all = long_all.merge(calendar[cal_cols], on="d", how="left")
    long_all = long_all.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long_all["date"] = pd.to_datetime(long_all["date"])

    # Integer day index for all ordering/splitting logic. Computed while `d` is
    # still a plain string, so downstream code never has to compare categoricals.
    day_map = {d: _day_index(d) for d in long_all["d"].unique()}
    long_all["day_idx"] = long_all["d"].map(day_map).astype("int32")

    # Downcast aggressively: at full M5 scale this frame is ~60M rows and the
    # default int64/object dtypes will exhaust memory. float32 supports the NaN
    # future rows; category dtypes collapse the repeated id and day strings.
    long_all["sales"] = long_all["sales"].astype("float32")
    long_all["d"] = long_all["d"].astype("category")
    for col in ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]:
        if col in long_all.columns:
            long_all[col] = long_all[col].astype("int8")
    if "year" in long_all.columns:
        long_all["year"] = long_all["year"].astype("int16")
    if "sell_price" in long_all.columns:
        long_all["sell_price"] = long_all["sell_price"].astype("float32")
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                "event_name_1", "event_type_1"]:
        if col in long_all.columns:
            long_all[col] = long_all[col].astype("category")
    return long_all, future_cols


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbosity": -1,
    "n_jobs": -1,
}


@dataclass
class LGBMForecaster:
    calendar: pd.DataFrame
    prices: pd.DataFrame
    cfg: Config
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    name: str = "lightgbm"
    booster_: lgb.Booster | None = None
    feature_cols_: list[str] = field(default_factory=list)

    def _prepare(self, train_wide, train_cols, horizon):
        long_all, future_cols = build_extended_long(
            train_wide, train_cols, horizon, self.calendar, self.prices
        )
        featured, feature_cols, categorical = build_features(long_all, self.cfg)
        return featured, feature_cols, categorical, future_cols

    def prepare_datasets(self, train_wide, train_cols, horizon) -> "PreparedData":
        """Build features once and return reusable train/valid datasets plus the
        future feature matrix. Sharing this across Optuna trials avoids rebuilding
        the (expensive) feature frame for every hyperparameter evaluation.
        """
        import time

        _t0 = time.time()
        featured, feature_cols, categorical, future_cols = self._prepare(
            train_wide, train_cols, horizon
        )
        print(
            f"    feature build: {len(featured):,} rows, {len(feature_cols)} features "
            f"in {time.time() - _t0:.0f}s",
            flush=True,
        )
        last_train_day = _day_index(train_cols[-1])
        is_future = featured["day_idx"] > last_train_day
        hist = featured[~is_future].dropna(subset=["sales"]).copy()

        # Time-based split for early stopping: the last `horizon` training days.
        valid_mask = hist["day_idx"] > (last_train_day - horizon)
        train_part = hist[~valid_mask]
        valid_part = hist[valid_mask]

        # feature_pre_filter must be off because the same Dataset is reused across
        # Optuna trials that vary min_child_samples (min_data_in_leaf); otherwise
        # LightGBM rejects a smaller value than the one it pre-filtered on.
        dataset_params = {"feature_pre_filter": False}
        train_set = lgb.Dataset(
            train_part[feature_cols],
            label=train_part["sales"],
            categorical_feature=categorical,
            free_raw_data=False,
            params=dataset_params,
        )
        valid_set = lgb.Dataset(
            valid_part[feature_cols],
            label=valid_part["sales"],
            categorical_feature=categorical,
            reference=train_set,
            free_raw_data=False,
            params=dataset_params,
        )
        future = featured[is_future].copy()
        return PreparedData(
            train_set=train_set,
            valid_set=valid_set,
            future=future,
            feature_cols=feature_cols,
            future_cols=future_cols,
        )

    def _train_booster(self, prepared: "PreparedData", params: dict[str, Any]) -> lgb.Booster:
        return lgb.train(
            params,
            prepared.train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[prepared.valid_set],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    def predict_from(self, prepared: "PreparedData", booster: lgb.Booster) -> pd.DataFrame:
        preds = booster.predict(
            prepared.future[prepared.feature_cols], num_iteration=booster.best_iteration
        )
        future = prepared.future.copy()
        future["prediction"] = np.clip(preds, 0, None)
        wide = future.pivot_table(
            index=ID_COLUMNS, columns="d", values="prediction", observed=True
        ).reset_index()
        return wide[ID_COLUMNS + prepared.future_cols]

    def fit_predict(
        self, train_wide, train_cols, horizon
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        prepared = self.prepare_datasets(train_wide, train_cols, horizon)
        self.feature_cols_ = prepared.feature_cols
        self.booster_ = self._train_booster(prepared, self.params)
        wide = self.predict_from(prepared, self.booster_)
        info = {
            "best_iteration": self.booster_.best_iteration,
            "n_features": len(prepared.feature_cols),
        }
        return wide, info

    def __call__(self, train_wide, train_cols, horizon):
        wide, _ = self.fit_predict(train_wide, train_cols, horizon)
        return wide

    def feature_importance(self) -> pd.DataFrame:
        if self.booster_ is None:
            raise RuntimeError("Model not trained yet.")
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_cols_,
                    "gain": self.booster_.feature_importance(importance_type="gain"),
                }
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


def tune_lightgbm(
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: Config,
    n_trials: int,
    horizon: int,
    timeout_minutes: float | None = None,
) -> tuple[dict[str, Any], float]:
    """Optuna search scored on WRMSSE over a single held-out window.

    Tuning uses the window immediately before the backtest windows so the final
    backtest stays untouched by the search. The objective is WRMSSE, matching the
    metric the project actually reports.

    The feature frame and the LightGBM datasets are built once and reused across
    every trial; only the booster is retrained per trial. Progress is logged after
    each trial, and an optional wall-clock timeout guarantees the search returns
    the best result found so far rather than running unbounded.
    """
    import sys
    import time

    import optuna

    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    n_windows = int(cfg.get("data.n_backtest_windows"))
    # Reserve the backtest windows at the very end; tune on the window just before.
    tune_valid_end = len(day_cols) - n_windows * horizon
    tune_train_cols = day_cols[: tune_valid_end - horizon]
    tune_valid_cols = day_cols[tune_valid_end - horizon : tune_valid_end]

    tune_train_wide = sales_wide[ID_COLUMNS + tune_train_cols].copy()
    tune_valid_wide = sales_wide[ID_COLUMNS + tune_valid_cols].copy()
    evaluator = WRMSSEEvaluator(tune_train_wide, tune_valid_wide, calendar, prices)

    # Build features and datasets a single time.
    print("Building tuning features (once) ...", flush=True)
    model = LGBMForecaster(calendar, prices, cfg)
    prepared = model.prepare_datasets(tune_train_wide, tune_train_cols, horizon)
    print(f"Feature build complete: {len(prepared.feature_cols)} features.", flush=True)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(DEFAULT_PARAMS)
        params.update(
            {
                "tweedie_variance_power": trial.suggest_float(
                    "tweedie_variance_power", 1.05, 1.9
                ),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            }
        )
        booster = model._train_booster(prepared, params)
        forecast = model.predict_from(prepared, booster)
        score, _ = evaluator.score(forecast)
        return score

    start = time.time()

    def log_callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        elapsed = time.time() - start
        value = trial.value if trial.value is not None else float("nan")
        print(
            f"  trial {trial.number + 1}/{n_trials}  WRMSSE={value:.4f}  "
            f"best={study.best_value:.4f}  elapsed={elapsed:.0f}s",
            flush=True,
        )
        sys.stdout.flush()

    timeout = None
    if timeout_minutes is None:
        timeout_minutes = cfg.get("models.optuna.timeout_minutes")
    if timeout_minutes:
        timeout = float(timeout_minutes) * 60.0

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[log_callback],
        show_progress_bar=False,
    )
    print(
        f"Tuning finished: {len(study.trials)} trials in {time.time() - start:.0f}s.",
        flush=True,
    )

    best = dict(DEFAULT_PARAMS)
    best.update(study.best_params)
    return best, study.best_value
