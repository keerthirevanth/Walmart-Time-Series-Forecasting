"""Provider-agnostic LLM client with rotation and fallback.

The insight code depends only on the small ``LLMProvider`` interface, so the
underlying model is swappable. Three concrete providers are supported:

- groq   : OpenAI-compatible chat completions (free tier, fast open models).
- gemini : Google Generative AI.
- echo   : no network, returns the assembled prompt. Used for tests and for
           running the pipeline end to end without any API key.

Because free tiers rate-limit aggressively, a ``RotatingProvider`` tries an
ordered list of (provider, model, key) entries and moves to the next one whenever
a call fails. Keys are read from the environment by name; the config never
contains the secrets themselves.
"""

from __future__ import annotations

import os
from typing import Protocol

from src.config import Config


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - interface
        ...


class EchoProvider:
    """Offline provider: returns a deterministic rendering of the prompt."""

    name = "echo"

    def complete(self, system: str, prompt: str) -> str:
        return (
            "[echo provider - no LLM was called]\n\n"
            f"SYSTEM:\n{system}\n\n"
            f"PROMPT:\n{prompt}"
        )


class GroqProvider:
    def __init__(self, model: str, api_key: str, max_tokens: int, temperature: float):
        from groq import Groq  # lazy

        self._client = Groq(api_key=api_key)
        self.model = model
        self.name = f"groq:{model}"
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()


class GeminiProvider:
    def __init__(self, model: str, api_key: str, max_tokens: int, temperature: float):
        import google.generativeai as genai  # lazy

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model
        self.name = f"gemini:{model}"
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, prompt: str) -> str:
        model = self._genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return resp.text.strip()


class RotatingProvider:
    """Try each provider in order; on failure, fall back to the next.

    ``providers`` is the ordered list of already-constructed providers. The
    ``name`` attribute reflects whichever one last served a successful call.
    """

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("RotatingProvider needs at least one provider.")
        self.providers = providers
        self.name = "rotating(" + ", ".join(p.name for p in providers) + ")"

    def complete(self, system: str, prompt: str) -> str:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                out = provider.complete(system, prompt)
                self.name = f"rotating->{provider.name}"
                return out
            except Exception as error:  # rate limit, quota, transient, etc.
                last_error = error
                print(
                    f"[rotate] {provider.name} failed ({type(error).__name__}: "
                    f"{error}); trying next provider ...",
                    flush=True,
                )
        raise RuntimeError(f"All providers failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Construction from config
# ---------------------------------------------------------------------------
def _build_entry(entry: dict, max_tokens: int, temperature: float) -> LLMProvider | None:
    """Build a single provider from a rotation entry, or None if unusable.

    An entry is skipped (returns None) when its key environment variable is not
    set or the provider SDK is not installed, so users only need to configure the
    providers they actually have keys for.
    """
    provider = entry["provider"].lower()
    if provider == "echo":
        return EchoProvider()

    key = os.environ.get(entry.get("key_env", ""))
    if not key:
        return None
    try:
        if provider == "groq":
            return GroqProvider(entry["model"], key, max_tokens, temperature)
        if provider == "gemini":
            return GeminiProvider(entry["model"], key, max_tokens, temperature)
    except Exception as error:  # SDK missing or client init failed
        print(f"[insight] skipping {provider}:{entry.get('model')} ({error})", flush=True)
        return None
    raise ValueError(f"Unknown provider in rotation entry: {provider!r}")


def get_provider(cfg: Config, override: str | None = None) -> LLMProvider:
    max_tokens = int(cfg.get("insight.max_tokens", 1200))
    temperature = float(cfg.get("insight.temperature", 0.2))
    mode = (override or cfg.get("insight.provider", "echo")).lower()

    if mode == "echo":
        return EchoProvider()

    if mode in ("groq", "gemini"):
        entry = {
            "provider": mode,
            "model": cfg.get(f"insight.{mode}_model"),
            "key_env": "GROQ_API_KEY" if mode == "groq" else "GEMINI_API_KEY",
        }
        built = _build_entry(entry, max_tokens, temperature)
        if built is None:
            raise RuntimeError(
                f"Provider '{mode}' requested but its API key env var is not set."
            )
        return built

    if mode == "rotating":
        rotation = cfg.get("insight.rotation") or []
        providers = [
            p
            for entry in rotation
            if (p := _build_entry(entry, max_tokens, temperature)) is not None
        ]
        if not providers:
            raise RuntimeError(
                "Rotation is configured but no usable providers were found. Set at "
                "least one of the API key environment variables referenced in "
                "config.yaml 'insight.rotation'."
            )
        return RotatingProvider(providers)

    raise ValueError(f"Unknown insight provider mode: {mode!r}")
