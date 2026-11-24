"""FinSentry Unit 4 - 15-question analyst evaluation.

``questions.yaml`` holds exactly 15 questions with KNOWN answers computed
INDEPENDENTLY of the analyst (hand-verified ground-truth values, not produced by
calling the analyst's own tools). ``run_eval`` drives all 15 through the agent,
compares answers to the known values, reports accuracy + failure cases, and logs
every tool call to ``reports/eval_tool_calls.jsonl``.
"""

from __future__ import annotations

__all__ = ["run_eval"]
