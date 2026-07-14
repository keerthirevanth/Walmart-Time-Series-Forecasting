"""Data quality checks run before any modelling.

The goal is to fail fast on structural problems (missing days, unexpected
duplicates, negative sales) rather than discovering them mid-training. Results
are printed and returned as a structured report so they can be logged.

Usage:
    python -m src.data.validate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import load_config


@dataclass
class ValidationReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks[name] = passed
        if detail:
            self.details[name] = detail

    @property
    def all_passed(self) -> bool:
        return all(self.checks.values())

    def render(self) -> str:
        lines = ["Data validation report", "-" * 40]
        for name, passed in self.checks.items():
            status = "PASS" if passed else "FAIL"
            lines.append(f"[{status}] {name}")
            if name in self.details:
                lines.append(f"        {self.details[name]}")
        lines.append("-" * 40)
        lines.append("Overall: " + ("PASS" if self.all_passed else "FAIL"))
        return "\n".join(lines)


def validate_long_frame(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()

    # 1. No duplicate (series, date) pairs.
    n_dupes = df.duplicated(subset=["id", "date"]).sum()
    report.add(
        "no_duplicate_series_date",
        n_dupes == 0,
        f"{n_dupes} duplicate (id, date) rows",
    )

    # 2. Every series covers a contiguous daily range with no gaps.
    span = df.groupby("id", observed=True)["date"].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days + 1
    n_gapped = int((expected != span["count"]).sum())
    report.add(
        "contiguous_daily_index",
        n_gapped == 0,
        f"{n_gapped} series have missing days",
    )

    # 3. Sales are non-negative integers.
    n_negative = int((df["sales"] < 0).sum())
    report.add("non_negative_sales", n_negative == 0, f"{n_negative} negative sales rows")

    # 4. Prices are positive where present (missing prices mean not-yet-listed items).
    priced = df["sell_price"].dropna()
    n_bad_price = int((priced <= 0).sum())
    report.add(
        "positive_prices_when_present",
        n_bad_price == 0,
        f"{n_bad_price} non-positive prices",
    )

    # 5. Report leading-zero share: early zeros often mean the item was not yet
    #    on sale rather than genuine zero demand. This is informational.
    zero_share = float((df["sales"] == 0).mean())
    report.add(
        "zero_sales_share_reported",
        True,
        f"{zero_share:.1%} of rows have zero sales (intermittent demand is expected)",
    )

    return report


def main() -> None:
    cfg = load_config()
    path = cfg.path("data.processed_dir") / "sales_long.parquet"
    if not Path(path).exists():
        raise SystemExit(f"Processed frame not found at {path}. Run src.data.load first.")

    df = pd.read_parquet(path)
    report = validate_long_frame(df)
    print(report.render())
    if not report.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
