# Data access

The project uses the **M5 Forecasting - Accuracy** competition dataset from
Kaggle. The data is real anonymised Walmart sales history and is not committed to
this repository.

## One-time setup

1. Create a Kaggle account and open the competition page:
   https://www.kaggle.com/competitions/m5-forecasting-accuracy
2. Accept the competition rules (required before the API will serve the files).
3. Create an API token: Kaggle account settings -> "Create New API Token". This
   downloads `kaggle.json`.
4. Make the token available to the CLI in one of two ways:
   - place `kaggle.json` at `~/.kaggle/kaggle.json` (Unix) or
     `C:\Users\<you>\.kaggle\kaggle.json` (Windows); or
   - set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Download

```bash
python -m src.data.download
```

This fetches and unzips the following files into `data/raw/`:

| File | Description |
|------|-------------|
| `calendar.csv` | Maps day ids to dates, weekday, month, events, SNAP flags. |
| `sell_prices.csv` | Weekly selling price per store-item. |
| `sales_train_evaluation.csv` | Daily unit sales, 1,941 days of history. |
| `sales_train_validation.csv` | Same but 28 days shorter (used by the public LB). |
| `sample_submission.csv` | Submission template (row ordering reference). |

## Notes on the labels

- `sales_train_evaluation.csv` carries 28 more days than the validation file. We
  use the evaluation file as the full history and hold out the final windows for
  backtesting locally, so the project never depends on the closed Kaggle
  leaderboard.
- Leading zeros in a series usually mean the item was not yet on sale rather than
  genuine zero demand. The scale used by WRMSSE and MASE ignores these leading
  zeros; the feature pipeline treats them accordingly.
