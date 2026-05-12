"""Streaming tool-call sieve.

This keeps possible tool-call markup out of streamed assistant text until it can
be parsed into OpenAI-compatible tool_calls. It intentionally starts small and
targets the formats Qwen web has been observed to emit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from qwen_proxy import toolcall


TOOL_MARKER_RE = re.compile(
    r"<\s*[\|｜]?\s*(?:DSML\s*[\|｜]\s*)?tool_calls\b|"
    r"</?\s*[\|｜]?\s*tool_call\s*[\|｜]?\s*>|"
    r"<\s*tool_call\b|"
    r"\{\s*\"tool_calls\"\s*:",
    re.IGNORECASE,
)

PARTIAL_HOLD = 64


@dataclass
class ToolStreamEvent:
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


class ToolStreamSieve:
    def __init__(self, tools: list[dict[str, Any]] | None = None):
        self.tools = tools
        self.buffer = ""
        self.tool_emitted = False

    def feed(self, chunk: str) -> list[ToolStreamEvent]:
        if not chunk:
            return []
        if self.tool_emitted:
            return []

        self.buffer += chunk
        return self._drain(final=False)

    def flush(self) -> list[ToolStreamEvent]:
        if self.tool_emitted:
            self.buffer = ""
            return []
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[ToolStreamEvent]:
        events: list[ToolStreamEvent] = []
        normalized = toolcall._normalize_markup_chars(self.buffer)
        calls = toolcall.parse_tool_calls(normalized, self.tools)
        if calls:
            prefix = _prefix_before_tool_marker(normalized)
            if prefix:
                events.append(ToolStreamEvent(content=prefix))
            events.append(ToolStreamEvent(tool_calls=calls))
            self.buffer = ""
            self.tool_emitted = True
            return events

        marker = TOOL_MARKER_RE.search(normalized)
        if marker:
            if marker.start() > 0:
                prefix = normalized[: marker.start()]
                self.buffer = self.buffer[marker.start():]
                events.append(ToolStreamEvent(content=prefix))
            if final:
                # Avoid exposing malformed tool markup as user-visible text.
                if not _looks_like_tool_markup(normalized):
                    events.append(ToolStreamEvent(content=normalized))
                self.buffer = ""
            return events

        if final:
            if self.buffer:
                events.append(ToolStreamEvent(content=self.buffer))
                self.buffer = ""
            return events

        safe_len = max(0, len(self.buffer) - PARTIAL_HOLD)
        if safe_len:
            safe = self.buffer[:safe_len]
            self.buffer = self.buffer[safe_len:]
            events.append(ToolStreamEvent(content=safe))
        return events


def _prefix_before_tool_marker(text: str) -> str:
    marker = TOOL_MARKER_RE.search(text)
    if not marker:
        return ""
    return text[: marker.start()]


def _looks_like_tool_markup(text: str) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in ("tool_calls", "tool_call", "dsml"))
