"""Feature engineering for the gradient-boosting model.

All features are horizon-safe: computed only from information available at the
forecast cutoff, so a single model can predict the whole 28-day horizon without
leakage. Lags are at least the horizon length, and rolling statistics are taken
on the sales series shifted by the same base lag.

The functions operate on a long frame (one row per series-day) that already
carries the calendar and price columns produced by ``src.data.load``. To produce
features for future dates, append placeholder rows for those dates (with NaN
sales) before calling ``build_features``; the lag/rolling machinery then fills
them in from the known history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "wday",
    "month",
    "event_name_1",
    "event_type_1",
]


def _add_calendar_features(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    df["day_of_month"] = df["date"].dt.day.astype(np.int8)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int16)
    df["is_weekend"] = df["wday"].isin([1, 2]).astype(np.int8)  # wday 1=Sat, 2=Sun
    cols += ["day_of_month", "week_of_year", "is_weekend", "wday", "month", "year"]

    # SNAP flag for the series' own state (the three snap columns are state-specific).
    snap = np.zeros(len(df), dtype=np.int8)
    for state in ["CA", "TX", "WI"]:
        col = f"snap_{state}"
        if col in df.columns:
            mask = (df["state_id"] == state).to_numpy()
            snap[mask] = df.loc[mask, col].to_numpy()
    df["snap"] = snap
    cols.append("snap")
    return cols


def _add_event_features(df: pd.DataFrame) -> list[str]:
    df["has_event"] = df["event_name_1"].notna().astype(np.int8)
    return ["has_event"]


def _add_price_features(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    grp = df.groupby("id", observed=True)["sell_price"]
    # Price relative to the item's own history, and short-term price change.
    # transform("mean"/"max") and groupby.pct_change are vectorised in C; a
    # per-group Python lambda here would dominate runtime on 20M+ rows.
    df["price_norm"] = (df["sell_price"] / grp.transform("mean")).astype(np.float32)
    df["price_max_ratio"] = (df["sell_price"] / grp.transform("max")).astype(np.float32)
    df["price_change"] = grp.pct_change(fill_method=None).astype(np.float32)
    cols += ["sell_price", "price_norm", "price_max_ratio", "price_change"]
    return cols


def _add_lag_features(df: pd.DataFrame, lags: list[int]) -> list[str]:
    cols: list[str] = []
    grp = df.groupby("id", observed=True)["sales"]
    for lag in lags:
        name = f"lag_{lag}"
        df[name] = grp.shift(lag).astype(np.float32)
        cols.append(name)
    return cols


def _add_rolling_features(
    df: pd.DataFrame, base_lag: int, windows: list[int], stats: list[str]
) -> list[str]:
    cols: list[str] = []
    df["_shifted"] = df.groupby("id", observed=True)["sales"].shift(base_lag)
    grp = df.groupby("id", observed=True)["_shifted"]
    # groupby.rolling runs in C, unlike transform(lambda ...) which loops in
    # Python once per series and is the main cost on the full dataset. The result
    # carries a (group, original_index) MultiIndex; dropping the group level
    # realigns it to the frame's own index.
    for window in windows:
        min_periods = max(1, window // 2)
        if "mean" in stats:
            name = f"rmean_{window}"
            rolled = grp.rolling(window, min_periods=min_periods).mean()
            df[name] = rolled.reset_index(level=0, drop=True).astype(np.float32)
            cols.append(name)
        if "std" in stats:
            name = f"rstd_{window}"
            rolled = grp.rolling(window, min_periods=min_periods).std()
            df[name] = rolled.reset_index(level=0, drop=True).astype(np.float32)
            cols.append(name)
    df.drop(columns=["_shifted"], inplace=True)
    return cols


def build_features(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return the frame with engineered features plus the feature/categorical lists.

    The input must be sorted-able by (id, date); it is sorted internally. Rows for
    future dates (NaN sales) are retained so their features can be predicted on.
    """
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    feature_cols: list[str] = []
    if cfg.get("features.calendar_features", True):
        feature_cols += _add_calendar_features(df)
    if cfg.get("features.encode_events", True):
        feature_cols += _add_event_features(df)
    if cfg.get("features.price_features", True):
        feature_cols += _add_price_features(df)

    feature_cols += _add_lag_features(df, cfg.get("features.lags"))
    feature_cols += _add_rolling_features(
        df,
        base_lag=int(cfg.get("features.rolling_base_lag")),
        windows=cfg.get("features.rolling_windows"),
        stats=cfg.get("features.rolling_stats"),
    )

    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for col in categorical:
        if col not in feature_cols:
            feature_cols.append(col)
        # LightGBM reads pandas 'category' dtype natively.
        if str(df[col].dtype) != "category":
            df[col] = df[col].astype("category")

    return df, feature_cols, categorical
