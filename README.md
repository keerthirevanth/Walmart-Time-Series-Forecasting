# Retail Demand Forecasting on M5

End-to-end demand forecasting on the full Walmart M5 dataset (30,490 store-item
series across three US states), built as a controlled benchmark under the
competition's official metric. Classical statistical baselines, an Optuna-tuned
LightGBM, and the Chronos time-series foundation model are all evaluated on the
same rolling-origin backtest; a grounded LLM insight layer then turns the results
into plain-language analysis.

**Headline result:** the tuned LightGBM reaches a WRMSSE of 0.648, about 28%
better than the strongest baseline, and is best on WRMSSE, MASE, and RMSE at once.
Chronos, evaluated zero-shot, has the best per-series MASE yet the worst
hierarchy-weighted WRMSSE - a scale-confirmed finding on the limits of zero-shot
foundation models for intermittent, hierarchical demand.

The aim is not a single leaderboard number but a defensible comparison: which
class of model earns its cost, where in the product hierarchy each one wins or
loses, and how uncertain the forecasts are.

## Problem

Forecast daily unit sales for the next 28 days across 30,490 store-item series
organised in a 12-level hierarchy (item, department, category, store, state and
their combinations). The series are highly intermittent: a large share of days
have zero sales, which rules out naive use of symmetric error objectives and
motivates the modelling choices below.

## Data

- **Source:** [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
  on Kaggle. Real Walmart sales history covering three US states (California,
  Texas, Wisconsin), roughly five years of daily data, with calendar events,
  SNAP benefit days, and weekly selling prices.
- **Access:** downloaded through the Kaggle API. The competition rules must be
  accepted once on the website before the API will serve the files. See
  [docs/DATA.md](docs/DATA.md).
- The raw files are not committed to the repository; only the code that produces
  the processed frames is.

## Approach

1. **Data pipeline** - download, reshape the wide sales table to a long tidy
   frame, merge calendar and prices, downcast dtypes, and run structural
   validation (contiguous daily index, no duplicates, non-negative sales).
2. **EDA** - quantify seasonality and, importantly, classify every series by its
   demand pattern (Syntetos-Boylan ADI / CV^2) to justify later modelling
   choices.
3. **Baselines** - naive, seasonal naive, moving average, and exponential
   smoothing, so every learned model is measured against an honest floor.
4. **Gradient boosting** - LightGBM with lag, rolling, calendar, and price
   features and a Tweedie objective suited to intermittent counts;
   hyperparameters tuned with Optuna.
5. **Foundation model** - Amazon Chronos-Bolt evaluated zero-shot (mean and
   median point forecasts), to measure what a pretrained time-series model buys
   over a task-specific one. Fine-tuning is noted as future work.
6. **Evaluation** - a single rolling-origin backtest scores every model on
   WRMSSE (primary), MASE and RMSE (point), and pinball loss (probabilistic),
   with a per-level breakdown.
7. **Insight layer** - an LLM turns the numerical forecasts and backtest
   diagnostics into plain-language explanations and answers questions about the
   forecast.

## Metric

The headline metric is **WRMSSE**, the official M5 metric: a dollar-sales
weighted average of scaled error across the 12 hierarchy levels. It is
implemented from first principles in `src/evaluation/metrics.py` and covered by
known-answer unit tests.

## Repository layout

```
config/            central YAML configuration for every run
src/
  config.py        config loader
  data/            download, load/reshape, validate
  features/        feature engineering (Phase 1)
  models/          baselines, LightGBM, foundation model (Phases 1-3)
  evaluation/      metrics (WRMSSE, MASE, pinball) and backtesting
  insight/         LLM insight layer: provider client, grounded context, analyst
  eda.py           exploratory analysis and figures
scripts/           runnable entry points (backtests, export, insight CLI)
tests/             unit tests for the metrics, pipeline, and insight layer
reports/           figures, backtest results, and the exported forecast
docs/              data access and methodology notes
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
# Optional (foundation model + LLM layer, ideally on a GPU box):
# pip install -r requirements-optional.txt
```

Configure Kaggle credentials (see [docs/DATA.md](docs/DATA.md)), then:

```bash
python -m src.data.download     # fetch the raw competition files
python -m src.data.load         # reshape to a cached long parquet
python -m src.data.validate     # structural data-quality checks
python -m src.eda               # summary statistics and figures
```

Experiment scope (a fast subset for iteration vs. the full 30,490 series) is
controlled entirely from `config/config.yaml`.

## Results

Rolling-origin backtest, 3 windows of 28 days, full M5 (all 30,490 series across
California, Texas, and Wisconsin). Lower is better; WRMSSE is the primary metric.

| Model | WRMSSE | MASE | RMSE | Notes |
|-------|--------|------|------|-------|
| **lightgbm (tuned)** | **0.648** | 1.704 | **2.198** | Global Tweedie model, Optuna-tuned; wins on all three metrics. |
| seasonal_naive | 0.894 | 1.851 | 2.732 | Strongest baseline on WRMSSE; captures the weekly cycle. |
| moving_average (28) | 1.077 | 1.647 | 2.256 | Flat forecast; competitive point accuracy on noisy bottom series. |
| croston_sba | 1.089 | 1.669 | 2.282 | Intermittent-demand baseline (most series are intermittent/lumpy). |
| naive | 1.549 | 2.012 | 3.111 | Last-value; weakest, as expected on weekly-seasonal data. |
| chronos-bolt (zero-shot) | 2.038 | **1.387** | 2.308 | Foundation model, no training; best MASE but worst WRMSSE (see below). |

The tuned LightGBM improves WRMSSE by roughly 28% over the strongest baseline and,
unlike any single baseline, is best on WRMSSE, MASE, and RMSE at once. The Optuna
search is itself informative: it settled on a Tweedie variance power of ~1.30 (a
compound-Poisson regime that fits zero-inflated counts) with heavy regularisation
(shallow trees, large minimum leaf size), the right response to sparse,
intermittent demand. Hyperparameters are saved to
`models/lightgbm_best_params.json`. The same experiment was first run on the
California subset (12,196 series), where LightGBM scored 0.602 and led by ~20%;
the lead widened at full scale.

The earlier gap between the WRMSSE and MASE/RMSE rankings among the baselines is
still informative: WRMSSE is dollar-weighted and rolled up the hierarchy, so it
rewards getting the high-volume seasonal series right, whereas squared-error point
metrics on the sparse bottom series can favour a flat forecast.

### Foundation model: an instructive result

Chronos-Bolt was evaluated zero-shot (no training on M5), scored with both the
mean and the median of its predictive distribution. The result is the most
interesting comparison in the project:

- Chronos has the **best MASE of any model (1.39)** and a competitive RMSE, so its
  per-series point forecasts are genuinely good.
- Yet it has the **worst WRMSSE (2.04)** - below even the naive forecast.
- Mean vs. median made essentially no difference, which ruled out the obvious
  explanation that the median of an intermittent distribution collapses to zero.

The cause is the hierarchy. WRMSSE aggregates forecasts up 12 levels and weights
them by dollar sales; the top levels are smooth, high-volume series whose scaled
error denominators are small. Small, same-direction biases across the 30,490
bottom series do not cancel when summed, so they surface as large errors at the
aggregate levels that WRMSSE weights most heavily. A model tuned for this data
with a Tweedie mean objective (LightGBM) preserves those aggregates; a general
pretrained model scored per series does not.

The takeaway is not that foundation models are bad, but that headline per-series
accuracy can hide poor hierarchical coherence, and the choice of metric decides
the winner. Fine-tuning Chronos on M5 and forecast reconciliation across the
hierarchy are the natural next steps and are left as future work.

### Backtest design and its validation

The primary results use three rolling 28-day windows. That choice was checked,
not assumed: `scripts/window_sensitivity.py` re-runs the baselines at 3, 5, and 6
windows (validation run on the California subset during development).

| Model | w=3 | w=5 | w=6 |
|-------|-----|-----|-----|
| seasonal_naive | 0.752 | 0.816 | 0.906 |
| moving_average | 1.090 | 1.112 | 1.137 |
| croston_sba | 1.100 | 1.106 | 1.148 |
| naive | 1.796 | 1.813 | 1.876 |

The best and worst models are invariant to the window count (Spearman rho = 1.0
between 3 and 6 windows); the only movement is a swap between two mid-pack
baselines that sit within 0.006 of each other. Absolute WRMSSE rises with more
windows because older origins cover harder periods, so three windows gives a
slightly optimistic figure. The ranking conclusions therefore hold; where an
absolute number is quoted for a final model it is also reported at six windows as
a conservative check.

## Insight layer

A natural-language analyst sits on top of the numerical results. It is grounded:
every figure it can cite is loaded from the backtest and forecast artifacts and
injected into the prompt, so it reports the real numbers rather than inventing
them. It is provider-agnostic (Groq or Gemini, plus an offline `echo` provider
that needs no key) so the model is swappable behind one interface.

To survive free-tier rate limits, the default mode is `rotating`: it tries an
ordered list of (provider, model, key) entries and falls back to the next
whenever a call fails. The list lives in `config/config.yaml` under
`insight.rotation` and references key environment variables by name; entries
whose key is unset are skipped, so you only configure the providers you have.
Keys are never stored in the repository.

First export the operational forecast (this also writes the per-series context
the analyst uses):

```bash
python -m scripts.export_forecast
```

Then, with an API key in the environment (`GROQ_API_KEY` or `GEMINI_API_KEY`):

```bash
pip install -r requirements-insight.txt
python -m scripts.run_insight summary                       # executive narrative
python -m scripts.run_insight ask "Which category carries the most risk?"
python -m scripts.run_insight explain FOODS_3_090_CA_1_evaluation
python -m scripts.run_insight summary --provider echo       # no key, prints the grounded prompt
```

## Tests

```bash
python -m pytest -q
```
