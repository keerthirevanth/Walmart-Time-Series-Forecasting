"""Interactive dashboard for the M5 demand forecasting project.

Lets a user pick a store-item, see its recent sales history and 28-day forecast,
and query the grounded LLM analyst for an explanation or a free-form question.

Run with:
    streamlit run app.py

Reads the artifacts produced by the pipeline (reports/forecast, reports/backtests)
and, for plotting history, the raw M5 sales in data/raw. API keys for the analyst
are loaded from a local .env / .env.txt file if present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config
from src.insight.analyst import build_analyst
from src.evaluation.metrics import ID_COLUMNS

st.set_page_config(page_title="M5 Demand Forecasting", layout="wide")

CFG = load_config()
HORIZON = int(CFG.get("data.horizon"))
HISTORY_DAYS = 120


# ---------------------------------------------------------------------------
# Setup and cached loaders
# ---------------------------------------------------------------------------
def _load_env() -> None:
    for name in (".env", ".env.txt"):
        p = Path(name)
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
            return


@st.cache_data(show_spinner="Loading forecast ...")
def load_forecast() -> pd.DataFrame | None:
    path = CFG.path("paths.forecast_dir") / "forecast.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner="Loading series context ...")
def load_series_context() -> pd.DataFrame | None:
    path = CFG.path("paths.forecast_dir") / "series_context.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner="Loading leaderboard ...")
def load_leaderboard() -> pd.DataFrame | None:
    path = CFG.path("paths.backtests_dir") / "leaderboard.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Loading sales history ...")
def load_history() -> tuple[pd.DataFrame, pd.Series] | None:
    """Wide sales indexed by id (downcast) plus a day->date lookup."""
    raw = CFG.path("data.raw_dir")
    sales_path = raw / "sales_train_evaluation.csv"
    cal_path = raw / "calendar.csv"
    if not sales_path.exists() or not cal_path.exists():
        return None
    sales = pd.read_csv(sales_path)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales[day_cols] = sales[day_cols].astype("int16")
    sales = sales.set_index("id")
    calendar = pd.read_csv(cal_path, usecols=["d", "date"])
    day_to_date = calendar.set_index("d")["date"]
    return sales, day_to_date


@st.cache_resource(show_spinner="Connecting to the analyst ...")
def get_analyst(provider: str):
    """Build the analyst for a provider; fall back to the offline echo provider."""
    try:
        return build_analyst(CFG, provider_override=provider), None
    except Exception as error:  # missing key, SDK, etc.
        return build_analyst(CFG, provider_override="echo"), str(error)


def build_chart_frame(series_id, sales, day_to_date, forecast_row):
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    hist_days = day_cols[-HISTORY_DAYS:]
    last_idx = int(hist_days[-1].split("_")[1])
    future_days = [f"d_{last_idx + i}" for i in range(1, HORIZON + 1)]

    hist_dates = pd.to_datetime(day_to_date.reindex(hist_days).values)
    future_dates = pd.to_datetime(day_to_date.reindex(future_days).values)

    hist_values = sales.loc[series_id, hist_days].to_numpy(dtype=float)
    f_cols = [c for c in forecast_row.index if c.startswith("F")]
    forecast_values = forecast_row[f_cols].to_numpy(dtype=float)

    frame = pd.DataFrame(index=pd.DatetimeIndex(list(hist_dates) + list(future_dates)))
    frame["Actual"] = list(hist_values) + [None] * HORIZON
    frame["Forecast"] = [None] * HISTORY_DAYS + list(forecast_values)
    # Bridge the gap so the forecast line connects to the last actual point.
    frame.iloc[HISTORY_DAYS - 1, frame.columns.get_loc("Forecast")] = hist_values[-1]
    return frame


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
_load_env()

st.title("M5 Demand Forecasting")
st.caption(
    "Store-item demand forecasts on the Walmart M5 dataset, with a grounded "
    "language-model analyst. Forecasts come from the tuned LightGBM model."
)

forecast = load_forecast()
series_ctx = load_series_context()
history = load_history()

if forecast is None or series_ctx is None:
    st.warning(
        "Forecast artifacts not found. Run `python -m scripts.export_forecast` first, "
        "then place `forecast.parquet` and `series_context.parquet` under "
        "`reports/forecast/`."
    )
    st.stop()

# -- sidebar: selection and provider ---------------------------------------
with st.sidebar:
    st.header("Select a series")
    categories = sorted(series_ctx["cat_id"].dropna().unique())
    cat = st.selectbox("Category", categories)
    stores = sorted(series_ctx.loc[series_ctx["cat_id"] == cat, "store_id"].unique())
    store = st.selectbox("Store", stores)
    items = sorted(
        series_ctx.loc[
            (series_ctx["cat_id"] == cat) & (series_ctx["store_id"] == store), "item_id"
        ].unique()
    )
    item = st.selectbox("Item", items)

    st.divider()
    st.header("Analyst")
    provider = st.selectbox(
        "Model provider",
        ["rotating", "groq", "gemini", "echo"],
        help="rotating falls back across providers on rate limits. echo needs no key.",
    )
    keys_present = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    st.caption(
        "API key detected." if keys_present else "No API key found; add one to .env "
        "for live analysis, or use the echo provider."
    )

row = series_ctx[
    (series_ctx["cat_id"] == cat)
    & (series_ctx["store_id"] == store)
    & (series_ctx["item_id"] == item)
]
if row.empty:
    st.info("No forecast for this selection.")
    st.stop()
series_id = row.iloc[0]["id"]
ctx = row.iloc[0]

# -- KPI row ----------------------------------------------------------------
with st.container(horizontal=True):
    st.metric("28-day forecast (units)", f"{ctx['forecast_total']:.0f}", border=True)
    st.metric("Avg daily forecast", f"{ctx['forecast_mean_28']:.2f}", border=True)
    st.metric("Recent daily avg (28d)", f"{ctx['recent_mean_28']:.2f}", border=True)
    st.metric("Zero-sales share", f"{ctx['zero_rate']:.0%}", border=True)

# -- forecast chart ---------------------------------------------------------
with st.container(border=True):
    st.subheader(f"History and forecast: {series_id}")
    if history is None:
        st.info(
            "Sales history not available (data/raw not found), so only the forecast "
            "totals are shown. The chart needs the raw M5 files under data/raw."
        )
    else:
        sales, day_to_date = history
        if series_id in sales.index:
            chart_df = build_chart_frame(series_id, sales, day_to_date, forecast.set_index("id").loc[series_id])
            st.line_chart(chart_df, color=["#1f77b4", "#d62728"])
        else:
            st.info("This series id was not found in the sales history file.")

# -- analyst tabs -----------------------------------------------------------
analyst, fallback_reason = get_analyst(provider)
if fallback_reason:
    st.sidebar.warning(f"Using echo provider: {fallback_reason}")

tab_explain, tab_ask, tab_leaderboard = st.tabs(
    ["Forecast explanation", "Ask the analyst", "Model comparison"]
)

with tab_explain:
    st.write("Generate a plain-language explanation of this item's forecast.")
    if st.button("Explain this forecast", type="primary"):
        with st.spinner("Analysing ..."):
            st.markdown(analyst.explain_series(series_id))

with tab_ask:
    st.write("Ask a grounded question about the forecasts and model results.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if question := st.chat_input("e.g. Which category carries the most forecast risk?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                answer = analyst.answer(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_leaderboard:
    leaderboard = load_leaderboard()
    if leaderboard is not None:
        st.subheader("Backtest leaderboard (lower WRMSSE is better)")
        st.dataframe(leaderboard, hide_index=True, width="stretch")
    else:
        st.info("leaderboard.csv not found under reports/backtests/.")
    if st.button("Generate executive summary"):
        with st.spinner("Writing summary ..."):
            st.markdown(analyst.executive_summary())
