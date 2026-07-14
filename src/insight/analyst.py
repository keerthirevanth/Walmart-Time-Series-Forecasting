"""The forecast analyst: narrative, grounded Q&A, and per-series explanation.

Each method assembles a prompt from the grounded context (real numbers only) and
delegates the language generation to the configured provider. The system prompt
constrains the model to the supplied figures so the output stays faithful to the
actual results.
"""

from __future__ import annotations

from src.config import Config
from src.insight.context import InsightContext, load_context
from src.insight.llm_client import LLMProvider, get_provider

SYSTEM_PROMPT = (
    "You are a demand-forecasting analyst reporting to a supply-chain team. "
    "You are given factual context blocks containing model backtest results, "
    "tuned hyperparameters, feature importances, and forecast aggregates for the "
    "Walmart M5 dataset (California stores). Rules: rely only on the numbers in "
    "the provided context; never invent figures; if the context does not support "
    "an answer, say so plainly. WRMSSE is the primary competition metric (lower is "
    "better) and is dollar-weighted across a 12-level product hierarchy. Write "
    "concisely and professionally, in plain language a business stakeholder can "
    "act on. Do not use emojis."
)


class ForecastAnalyst:
    def __init__(self, cfg: Config, provider: LLMProvider, context: InsightContext):
        self.cfg = cfg
        self.provider = provider
        self.context = context

    # -- capability 1: executive narrative --------------------------------
    def executive_summary(self) -> str:
        prompt = (
            "Write an executive summary (roughly 200-300 words) of the demand "
            "forecasting results below. Cover: which model performed best on WRMSSE "
            "and by how much versus the baselines; the notable finding about the "
            "foundation model if present in the data; what the tuned hyperparameters "
            "imply about the demand; and the headline forecast aggregates. End with "
            "one or two concrete recommendations for the supply-chain team.\n\n"
            "CONTEXT:\n" + self.context.render_overview()
        )
        return self.provider.complete(SYSTEM_PROMPT, prompt)

    # -- capability 2: grounded question answering ------------------------
    def answer(self, question: str) -> str:
        prompt = (
            "Answer the question using only the context below. If the numbers "
            "needed are not present, say what is missing.\n\n"
            f"QUESTION: {question}\n\n"
            "CONTEXT:\n" + self.context.render_overview()
            + "\n\nHIGHEST-VOLUME SERIES:\n" + self.context.top_series(10)
        )
        return self.provider.complete(SYSTEM_PROMPT, prompt)

    # -- capability 3: per-series driver explanation ----------------------
    def explain_series(self, series_id: str) -> str:
        series_block = self.context.lookup_series(series_id)
        prompt = (
            "Explain the 28-day demand forecast for the series below to a category "
            "manager. Use the series' recent behaviour, its category and store, and "
            "the model's top features to justify the forecast. Note any risk (for "
            "example intermittent demand or a recent shift). Keep it under 150 "
            "words.\n\n"
            f"SERIES:\n{series_block}\n\n"
            "MODEL FEATURE IMPORTANCES (global):\n"
            + (
                self.context.feature_importance.head(10).to_string(index=False)
                if self.context.feature_importance is not None
                else "not available"
            )
        )
        return self.provider.complete(SYSTEM_PROMPT, prompt)


def build_analyst(cfg: Config, provider_override: str | None = None) -> ForecastAnalyst:
    context = load_context(cfg)
    provider = get_provider(cfg, override=provider_override)
    return ForecastAnalyst(cfg, provider, context)
