"""Structured JSON logging for qwen_proxy diagnostics.

All tool-related events (parse success, sieve drop, recovery, retry, etc.)
are logged as single-line JSON objects to stderr for easy grep/jq analysis.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def log_event(event_type: str, **fields: Any) -> None:
    """Log a structured JSON event to stderr.

    Args:
        event_type: One of the predefined event types:
            - tool_parse_ok: DSML/XML parse succeeded
            - tool_parse_broken_dsml: regex extraction from malformed DSML
            - tool_parse_legacy: legacy <|tool_call|> tag parse
            - tool_parse_json: JSON tool_calls parse
            - sieve_drop: markup detected but parse failed
            - tool_recovery: context-based inference recovery
            - tool_miss: all parsing paths failed
            - retry_attempt: retrying with simplified prompt
            - retry_ok: retry succeeded
            - retry_fail: retry also failed
            - prompt_truncate: conversation history truncated
            - prompt_info: prompt length / message count info
        **fields: Arbitrary key-value pairs for the event payload.
    """
    entry: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event_type,
    }
    for key, value in fields.items():
        if isinstance(value, str) and len(value) > 1000:
            entry[key] = value[:1000] + f"... ({len(value)} total)"
        else:
            entry[key] = value
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        line = json.dumps({"ts": entry["ts"], "event": event_type, "error": "unserializable fields"})
    print(line, file=sys.stderr, flush=True)
