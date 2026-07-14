"""Exploratory data analysis for the M5 demand data.

Produces a printed summary (safe to paste into notes) and a set of figures under
reports/figures. The analysis is deliberately geared towards the modelling
decisions that follow: the intermittency classification, for example, motivates
the choice of a Tweedie objective and Croston-style baselines for slow-moving
items.

Usage:
    python -m src.eda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: safe on a server without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config


def _load(cfg) -> pd.DataFrame:
    path = cfg.path("data.processed_dir") / "sales_long.parquet"
    if not Path(path).exists():
        raise SystemExit(f"Processed frame not found at {path}. Run src.data.load first.")
    return pd.read_parquet(path)


def classify_intermittency(df: pd.DataFrame) -> pd.DataFrame:
    """Syntetos-Boylan classification per series via ADI and CV^2.

    ADI = average interval between non-zero demands.
    CV^2 = squared coefficient of variation of the non-zero demand sizes.
    Cut-offs (ADI=1.32, CV^2=0.49) split series into smooth, erratic,
    intermittent, and lumpy demand patterns.
    """
    records = []
    for series_id, group in df.groupby("id", observed=True):
        sales = group["sales"].to_numpy()
        nz = sales[sales > 0]
        if nz.size == 0:
            records.append((series_id, np.nan, np.nan, "no_sales"))
            continue
        adi = len(sales) / nz.size
        cv2 = (nz.std() / nz.mean()) ** 2 if nz.mean() > 0 else np.nan

        if adi < 1.32 and cv2 < 0.49:
            label = "smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            label = "intermittent"
        elif adi < 1.32 and cv2 >= 0.49:
            label = "erratic"
        else:
            label = "lumpy"
        records.append((series_id, adi, cv2, label))

    return pd.DataFrame(records, columns=["id", "adi", "cv2", "pattern"])


def print_summary(df: pd.DataFrame, patterns: pd.DataFrame) -> None:
    print("=" * 60)
    print("M5 demand data - EDA summary")
    print("=" * 60)
    print(f"Rows                : {len(df):,}")
    print(f"Series (bottom level): {df['id'].nunique():,}")
    print(f"Date range          : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Days per series      : {df.groupby('id', observed=True).size().iloc[0]:,}")
    print()
    print("Sales distribution:")
    print(df["sales"].describe().to_string())
    print(f"Zero-sales share    : {(df['sales'] == 0).mean():.1%}")
    print(f"Missing sell_price   : {df['sell_price'].isna().mean():.1%} of rows")
    print()
    print("Demand pattern mix (Syntetos-Boylan):")
    print(patterns["pattern"].value_counts().to_string())
    print()
    print("Sales share by category:")
    print(
        df.groupby("cat_id", observed=True)["sales"].sum().sort_values(ascending=False).to_string()
    )
    print()
    print("Sales share by state:")
    print(
        df.groupby("state_id", observed=True)["sales"].sum().sort_values(ascending=False).to_string()
    )


def make_figures(df: pd.DataFrame, patterns: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall daily sales with a 28-day moving average.
    daily = df.groupby("date", observed=True)["sales"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(daily.index, daily.values, linewidth=0.6, alpha=0.5, label="daily")
    plt.plot(daily.index, daily.rolling(28).mean(), color="black", label="28d moving avg")
    plt.title("Total daily unit sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "01_total_daily_sales.png", dpi=130)
    plt.close()

    # 2. Day-of-week seasonality.
    dow = df.groupby("wday", observed=True)["sales"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(dow.index.astype(str), dow.values)
    plt.title("Average sales by day of week (1=Sat ... 7=Fri)")
    plt.tight_layout()
    plt.savefig(fig_dir / "02_dow_seasonality.png", dpi=130)
    plt.close()

    # 3. Monthly seasonality.
    monthly = df.groupby("month", observed=True)["sales"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(monthly.index.astype(str), monthly.values)
    plt.title("Average sales by month")
    plt.tight_layout()
    plt.savefig(fig_dir / "03_month_seasonality.png", dpi=130)
    plt.close()

    # 4. Intermittency scatter (ADI vs CV^2).
    valid = patterns.dropna(subset=["adi", "cv2"])
    plt.figure(figsize=(6, 5))
    for label, sub in valid.groupby("pattern"):
        plt.scatter(sub["adi"], sub["cv2"], s=4, alpha=0.4, label=label)
    plt.axvline(1.32, color="grey", linestyle="--", linewidth=0.8)
    plt.axhline(0.49, color="grey", linestyle="--", linewidth=0.8)
    plt.xlabel("ADI (average demand interval)")
    plt.ylabel("CV^2 of non-zero demand")
    plt.title("Demand pattern classification")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(fig_dir / "04_intermittency.png", dpi=130)
    plt.close()

    print(f"Figures written to {fig_dir}")


def main() -> None:
    cfg = load_config()
    df = _load(cfg)
    patterns = classify_intermittency(df)
    print_summary(df, patterns)
    make_figures(df, patterns, cfg.path("paths.figures_dir"))


if __name__ == "__main__":
    main()
