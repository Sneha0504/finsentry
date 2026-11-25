"""The two interchangeable analyst "brains" that drive the identical tool loop.

* :class:`OfflineProvider` - deterministic rule/intent planner (see
  :mod:`finsentry.analyst.planner`). No API key, no randomness, no wall-clock.
  Used for every committed/reproducible number.
* :class:`GroqProvider` - the live Groq chat-completions tool-calling loop using
  the installed ``groq`` SDK (OpenAI-compatible ``tools=[...]`` schema). Opt-in via
  ``GROQ_API_KEY`` / ``--provider groq``. It CANNOT run without a key; it degrades
  gracefully to the offline provider when no key is set or the model errors.

Both return the same :class:`AssistantTurn` shape (final ``content`` or a list of
:class:`ToolCall`) so :mod:`finsentry.analyst.agent` drives them identically.

Provider selection (see :func:`select_provider`): explicit ``provider`` arg wins;
otherwise ``GROQ_API_KEY`` present => groq, absent => offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from finsentry.config import Config


# --------------------------------------------------------------------------- #
# Shared turn / tool-call types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    """One model turn: a final answer (``content``) OR a batch of tool calls."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Message helpers (OpenAI/Groq message dict format is shared by both providers)
# --------------------------------------------------------------------------- #
def _last_user_question(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _tool_results_after_last_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parsed tool-result dicts that appear after the last user message, in order."""
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = i
    results: list[dict[str, Any]] = []
    for msg in messages[last_user_idx + 1:]:
        if msg.get("role") == "tool":
            try:
                results.append(json.loads(msg.get("content") or "{}"))
            except json.JSONDecodeError:
                results.append({"ok": False, "error": "unparseable tool result"})
    return results


# --------------------------------------------------------------------------- #
# Offline provider (deterministic planner)
# --------------------------------------------------------------------------- #
class OfflineProvider:
    """Deterministic planner brain. Drives the real tools via intent routing."""

    name = "offline"

    def __init__(self, cfg: Config, default_year: int | None = None) -> None:
        self.cfg = cfg
        self.default_year = default_year or cfg.start_date.year

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        from finsentry.analyst.planner import route

        question = _last_user_question(messages)
        plan = route(question, self.default_year)
        prior_results = _tool_results_after_last_user(messages)

        # Compose the final answer once the planned tools have run (single round),
        # or immediately for an unrouted (no-tool) question.
        if not plan.calls or prior_results:
            return AssistantTurn(content=plan.answer(prior_results), tool_calls=[])

        calls = [
            ToolCall(id=f"call_{plan.intent}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(plan.calls)
        ]
        return AssistantTurn(content=None, tool_calls=calls)


# --------------------------------------------------------------------------- #
# Groq provider (live LLM tool-calling loop)
# --------------------------------------------------------------------------- #
class GroqProvider:
    """Live Groq chat-completions tool-calling brain (OpenAI-compatible schema).

    Uses the installed ``groq`` SDK: ``client.chat.completions.create(model=...,
    messages=..., tools=..., tool_choice="auto")`` and reads
    ``resp.choices[0].message`` for ``.content`` and ``.tool_calls`` (each with
    ``.id``, ``.function.name``, ``.function.arguments`` as a JSON string) - the
    same wire shape OpenAI uses. ``temperature=0`` + ``seed`` are set for
    best-effort stability, but LLM output is NOT guaranteed byte-reproducible, which
    is exactly why the committed numbers use the offline provider.
    """

    name = "groq"

    def __init__(self, cfg: Config, model: str, api_key: str, temperature: float = 0.0) -> None:
        from groq import Groq  # imported here so offline runs need no groq import path

        self.cfg = cfg
        self.model = model
        self.temperature = temperature
        self.client = Groq(api_key=api_key)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.temperature,
            seed=0,
            max_tokens=1024,
        )
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return AssistantTurn(content=msg.content, tool_calls=tool_calls)


# --------------------------------------------------------------------------- #
# Selection + graceful degradation
# --------------------------------------------------------------------------- #
def resolve_provider_name(provider: str | None) -> str:
    """Resolve the provider NAME: explicit arg wins; else GROQ_API_KEY present => groq."""
    if provider in ("offline", "groq"):
        return provider
    if provider not in (None, "auto"):
        raise ValueError(f"unknown provider '{provider}' (use 'offline' or 'groq')")
    return "groq" if os.environ.get("GROQ_API_KEY") else "offline"


def select_provider(
    cfg: Config,
    provider: str | None = None,
    verbose: bool = False,
) -> tuple[Any, str, list[str]]:
    """Build a provider instance, degrading to offline if Groq is unavailable.

    Returns ``(provider_instance, resolved_name, notes)``. ``notes`` records any
    degradation so callers can surface it.
    """
    # Load a local .env (if present) so GROQ_API_KEY can live there.
    try:
        from dotenv import load_dotenv

        load_dotenv(cfg.project_root / ".env")
    except Exception:  # pragma: no cover - dotenv optional
        pass

    notes: list[str] = []
    want = resolve_provider_name(provider)

    if want == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        model = str(cfg.raw.get("analyst", {}).get("groq_model", "openai/gpt-oss-20b"))
        if not api_key:
            notes.append("GROQ_API_KEY not set -> degraded to offline provider")
            if verbose:
                print("  [provider] " + notes[-1], flush=True)
            return OfflineProvider(cfg), "offline", notes
        try:
            prov = GroqProvider(cfg, model=model, api_key=api_key)
            if verbose:
                print(f"  [provider] groq ({model})", flush=True)
            return prov, "groq", notes
        except Exception as exc:  # import/construction failure -> degrade
            notes.append(f"Groq init failed ({exc!r}) -> degraded to offline provider")
            if verbose:
                print("  [provider] " + notes[-1], flush=True)
            return OfflineProvider(cfg), "offline", notes

    if verbose:
        print("  [provider] offline (deterministic planner)", flush=True)
    return OfflineProvider(cfg), "offline", notes
