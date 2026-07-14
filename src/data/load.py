"""Load the raw M5 files, apply the configured subset, and reshape to long format.

The competition ships the sales history in a wide layout (one column per day).
For feature engineering and modelling a long ("tidy") layout is far more
convenient: one row per (series, day). This module handles the reshape, merges
the calendar and price tables, applies dtype downcasting to keep memory in
check, and caches the result as Parquet.

Usage:
    python -m src.data.load
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _read_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    # The evaluation file carries 28 more days of history than the validation file.
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    return calendar, prices, sales


def _apply_subset(sales: pd.DataFrame, cfg: Config, seed: int) -> pd.DataFrame:
    if not cfg.get("subset.enabled", False):
        return sales

    filters = {
        "state_id": cfg.get("subset.states"),
        "cat_id": cfg.get("subset.categories"),
        "store_id": cfg.get("subset.stores"),
    }
    for column, allowed in filters.items():
        if allowed:
            sales = sales[sales[column].isin(allowed)]

    max_items = cfg.get("subset.max_items")
    if max_items and sales["item_id"].nunique() > max_items:
        rng = np.random.default_rng(seed)
        keep = rng.choice(sales["item_id"].unique(), size=max_items, replace=False)
        sales = sales[sales["item_id"].isin(keep)]

    return sales.reset_index(drop=True)


def _melt_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    long_df = sales.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )
    long_df["sales"] = long_df["sales"].astype(np.int32)
    return long_df


def _merge_calendar_and_prices(
    long_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    calendar_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    long_df = long_df.merge(calendar[calendar_cols], on="d", how="left")
    long_df = long_df.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    long_df["date"] = pd.to_datetime(long_df["date"])
    return long_df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    categoricals = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    for col in categoricals:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]:
        if col in df.columns:
            df[col] = df[col].astype(np.int8)
    if "year" in df.columns:
        df["year"] = df["year"].astype(np.int16)
    if "sell_price" in df.columns:
        df["sell_price"] = df["sell_price"].astype(np.float32)
    return df


def load_wide(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the (subset) wide sales frame plus the calendar and price tables.

    The wide layout (one column per ``d_*`` day) is what the WRMSSE evaluator and
    the backtest harness consume directly, so this avoids an extra long->wide
    pivot for those code paths.
    """
    raw_dir = cfg.path("data.raw_dir")
    seed = int(cfg.get("project.random_seed", 42))
    calendar, prices, sales = _read_raw(raw_dir)
    sales = _apply_subset(sales, cfg, seed)
    return sales.reset_index(drop=True), calendar, prices


def build_long_frame(cfg: Config) -> pd.DataFrame:
    raw_dir = cfg.path("data.raw_dir")
    seed = int(cfg.get("project.random_seed", 42))

    calendar, prices, sales = _read_raw(raw_dir)
    sales = _apply_subset(sales, cfg, seed)
    long_df = _melt_to_long(sales)
    long_df = _merge_calendar_and_prices(long_df, calendar, prices)
    long_df = _downcast(long_df)
    long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)
    return long_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Reshape M5 to a long, cached frame.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output parquet path. Defaults to <processed_dir>/sales_long.parquet",
    )
    args = parser.parse_args()

    cfg = load_config()
    long_df = build_long_frame(cfg)

    processed_dir = cfg.path("data.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else processed_dir / "sales_long.parquet"
    long_df.to_parquet(out_path, index=False)

    n_series = long_df["id"].nunique()
    print(f"Wrote {len(long_df):,} rows for {n_series:,} series to {out_path}")
    print(f"Date range: {long_df['date'].min().date()} to {long_df['date'].max().date()}")
    mem_mb = long_df.memory_usage(deep=True).sum() / 1e6
    print(f"In-memory size: {mem_mb:,.1f} MB")


if __name__ == "__main__":
    main()
