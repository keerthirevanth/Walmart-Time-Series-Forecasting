"""Command-line interface to the forecast analyst (LLM insight layer).

Examples:
    python -m scripts.run_insight summary
    python -m scripts.run_insight ask "Which category carries the most forecast risk?"
    python -m scripts.run_insight explain FOODS_3_090_CA_1_evaluation
    python -m scripts.run_insight summary --provider gemini
    python -m scripts.run_insight summary --provider echo    # no API key needed

Set the API key in the environment first, e.g. GROQ_API_KEY or GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.insight.analyst import build_analyst


def main() -> None:
    # A shared parent lets --provider appear before or after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provider",
        default=None,
        help="Override the configured provider: groq | gemini | echo.",
    )

    parser = argparse.ArgumentParser(description="Forecast analyst (LLM insight layer).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", parents=[common], help="Generate the executive summary.")

    ask_p = sub.add_parser("ask", parents=[common], help="Ask a grounded question.")
    ask_p.add_argument("question", help="The question to answer.")

    explain_p = sub.add_parser(
        "explain", parents=[common], help="Explain the forecast for one series."
    )
    explain_p.add_argument("series_id", help="Series id (or a substring, e.g. an item id).")

    args = parser.parse_args()
    cfg = load_config()
    analyst = build_analyst(cfg, provider_override=args.provider)

    print(f"[provider: {analyst.provider.name}]\n")
    if args.command == "summary":
        print(analyst.executive_summary())
    elif args.command == "ask":
        print(analyst.answer(args.question))
    elif args.command == "explain":
        print(analyst.explain_series(args.series_id))


if __name__ == "__main__":
    main()
