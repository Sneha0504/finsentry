"""FinSentry Unit 4 - AI analyst agent.

An LLM tool-calling loop over three *real* tools (read-only SQL, the existing
anomaly check, and a deterministic chart renderer). Two interchangeable
"brains" drive the identical loop and the identical tools:

* :class:`~finsentry.analyst.llm.OfflineProvider` - a deterministic, rule/intent
  based planner (no API key, no randomness, no wall-clock) used for the
  committed/reproducible numbers (weekly review, number verification, 15-question
  eval).
* :class:`~finsentry.analyst.llm.GroqProvider` - the live Groq chat-completions
  tool-calling loop (opt-in via ``GROQ_API_KEY`` / ``--provider groq``).

Both compose their final answers *from real tool results* - never hardcoded.
"""

from __future__ import annotations

__all__ = [
    "tools",
    "llm",
    "agent",
]
