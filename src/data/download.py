"""Download the M5 Forecasting - Accuracy dataset from Kaggle.

Requires the Kaggle API credentials to be available, either through
``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``
environment variables. The competition rules must be accepted once on the
Kaggle website before the API will serve the files.

Usage:
    python -m src.data.download
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import load_config

EXPECTED_FILES = [
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sample_submission.csv",
]


def _all_present(raw_dir: Path) -> bool:
    return all((raw_dir / name).exists() for name in EXPECTED_FILES)


def download(raw_dir: Path, competition: str, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    if _all_present(raw_dir) and not force:
        print(f"All expected files already present in {raw_dir}; skipping download.")
        return

    archive = raw_dir / f"{competition}.zip"
    print(f"Downloading competition '{competition}' into {raw_dir} ...")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition,
                "-p",
                str(raw_dir),
            ],
            check=True,
        )
    except FileNotFoundError:
        sys.exit(
            "The 'kaggle' CLI was not found. Install it with 'pip install kaggle' "
            "and configure your API token."
        )
    except subprocess.CalledProcessError as error:
        sys.exit(
            f"Kaggle download failed (exit code {error.returncode}). Confirm you have "
            "accepted the competition rules and that your API token is valid."
        )

    if archive.exists():
        print(f"Extracting {archive.name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        archive.unlink()

    missing = [name for name in EXPECTED_FILES if not (raw_dir / name).exists()]
    if missing:
        sys.exit(f"Download completed but these files are missing: {missing}")
    print("Download complete. Files available:")
    for name in EXPECTED_FILES:
        size_mb = (raw_dir / name).stat().st_size / 1e6
        print(f"  {name:32s} {size_mb:8.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the M5 dataset from Kaggle.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()

    cfg = load_config()
    download(
        raw_dir=cfg.path("data.raw_dir"),
        competition=cfg.get("data.kaggle_competition"),
        force=args.force,
    )


if __name__ == "__main__":
    main()
