"""The tool-calling loop + per-call logging (provider-agnostic).

:func:`run_agent` hands the question + tool schemas to a provider, receives tool
calls, executes the REAL tools (:mod:`finsentry.analyst.tools`), feeds the results
back, and iterates (capped by ``max_iterations``) until the provider returns a
final text answer. EVERY tool call is logged (run id, iteration, tool, arguments,
result summary) - to a JSONL file when a logger is supplied, and always on the
returned :class:`AgentResult`. Offline-mode content is deterministic (no clock
unless a fixed ``run_ts`` is passed).

If a live Groq run raises mid-loop (model error), the loop degrades gracefully:
it rebuilds the offline provider and re-answers the question deterministically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from finsentry.analyst.llm import AssistantTurn, OfflineProvider
from finsentry.analyst.tools import TOOL_SCHEMAS, execute_tool, result_detail, result_summary
from finsentry.config import Config

SYSTEM_PROMPT = (
    "You are FinSentry's finance data analyst. Answer questions about the finance "
    "data warehouse by calling the provided tools: run_sql (read-only SQL over the "
    "`analytics` schema), run_anomaly_check (statistical anomaly detector), and "
    "create_chart (render a PNG). Always ground every number in a tool result - "
    "never invent figures. When you have the answer, reply with a short, direct "
    "sentence that states the value(s) in plain numbers (money as $NNNN.NN, no "
    "thousands separators; dates as YYYY-MM-DD)."
)


@dataclass
class AgentResult:
    question: str
    answer: str
    provider: str
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [tc["tool"] for tc in self.tool_calls]


class JsonlLogger:
    """Append-only JSONL writer with deterministic (sorted-key) records."""

    def __init__(self, path: str | Path, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    """Serialise an assistant turn with tool calls into OpenAI/Groq message form."""
    return {
        "role": "assistant",
        "content": turn.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, sort_keys=True),
                },
            }
            for tc in turn.tool_calls
        ],
    }


def run_agent(
    question: str,
    cfg: Config,
    provider: Any,
    *,
    run_id: str = "adhoc",
    max_iterations: int = 6,
    logger: JsonlLogger | None = None,
    run_ts: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> AgentResult:
    """Drive one question through the tool-calling loop; return the answer + logs."""
    tools = tools or TOOL_SCHEMAS
    prov = provider
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_logs: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    iterations = 0

    it = 0
    while it < max_iterations:
        iterations = it + 1
        try:
            turn: AssistantTurn = prov.complete(messages, tools)
        except Exception as exc:  # live-provider failure -> graceful degradation
            if getattr(prov, "name", "") == "groq":
                prov = OfflineProvider(cfg)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                tool_logs, tool_results = [], []
                it = 0
                continue
            raise RuntimeError(f"provider failed: {exc}") from exc

        if turn.tool_calls:
            messages.append(_assistant_message(turn))
            for tc in turn.tool_calls:
                result = execute_tool(tc.name, tc.arguments, cfg)
                tool_results.append(result)
                record: dict[str, Any] = {
                    "run_id": run_id,
                    "iteration": it,
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "result_summary": result_summary(result),
                    "result": result_detail(result),
                    "ok": bool(result.get("ok", False)),
                }
                if run_ts is not None:
                    record["run_ts"] = run_ts
                tool_logs.append(record)
                if logger is not None:
                    logger.log(record)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(result, sort_keys=True),
                    }
                )
            it += 1
            continue

        return AgentResult(
            question=question,
            answer=turn.content or "",
            provider=getattr(prov, "name", "unknown"),
            iterations=iterations,
            tool_calls=tool_logs,
            tool_results=tool_results,
            messages=messages,
        )

    return AgentResult(
        question=question,
        answer="(max iterations reached without a final answer)",
        provider=getattr(prov, "name", "unknown"),
        iterations=iterations,
        tool_calls=tool_logs,
        tool_results=tool_results,
        messages=messages,
    )
