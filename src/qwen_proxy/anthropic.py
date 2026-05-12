"""Minimal Anthropic Messages adapter for agent runners.

It maps Anthropic `/v1/messages` requests to the proxy's OpenAI-compatible
internal shape, then maps OpenAI chat responses back to Anthropic message/SSE
shapes. The goal is Claude Code-style tool loops, not full Anthropic parity.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def is_messages_path(path: str) -> bool:
    return path in {"/v1/messages", "/anthropic/v1/messages", "/messages"}


def is_models_path(path: str) -> bool:
    return path in {"/anthropic/v1/models"}


def is_count_tokens_path(path: str) -> bool:
    return path in {"/v1/messages/count_tokens", "/anthropic/v1/messages/count_tokens"}


def to_openai_request(req: dict[str, Any], default_model: str) -> dict[str, Any]:
    tools = [_anthropic_tool_to_openai(tool) for tool in req.get("tools") or [] if isinstance(tool, dict)]
    messages = []
    system = req.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        system_text = _content_to_text(system)
        if system_text.strip():
            messages.append({"role": "system", "content": system_text})

    for msg in req.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        messages.extend(_anthropic_message_to_openai(msg))

    out: dict[str, Any] = {
        "model": req.get("model") or default_model,
        "messages": messages,
        "stream": bool(req.get("stream")),
    }
    if tools:
        out["tools"] = tools
    tool_choice = _anthropic_tool_choice_to_openai(req.get("tool_choice"))
    if tool_choice is not None:
        out["tool_choice"] = tool_choice
    return out


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or tool.get("inputSchema") or tool.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        },
    }


def _anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice in {"auto", "none", "required"}:
            return choice
        return None
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind in {"any", "required"}:
        return "required"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def _anthropic_message_to_openai(msg: dict[str, Any]) -> list[dict[str, Any]]:
    role = msg.get("role")
    content = msg.get("content", "")
    if role == "assistant":
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in _as_blocks(content):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                })
        out = {"role": "assistant", "content": "\n".join(part for part in text_parts if part)}
        if tool_calls:
            out["content"] = None
            out["tool_calls"] = tool_calls
        return [out]

    if role == "user":
        out: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in _as_blocks(content):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "tool_result":
                if text_parts:
                    out.append({"role": "user", "content": "\n".join(text_parts)})
                    text_parts = []
                out.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _content_to_text(block.get("content", "")),
                })
        if text_parts or not out:
            out.append({"role": "user", "content": "\n".join(text_parts) if text_parts else _content_to_text(content)})
        return out

    if role == "system":
        return [{"role": "system", "content": _content_to_text(content)}]
    return [{"role": "user", "content": _content_to_text(content)}]


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [{"type": "text", "text": _content_to_text(content)}]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def from_openai_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict[str, Any]] = []
    if msg.get("content"):
        content_blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        try:
            input_obj = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            input_obj = {"arguments": args}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "name": fn.get("name", ""),
            "input": input_obj or {},
        })
    stop_reason = "tool_use" if msg.get("tool_calls") else "end_turn"
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": (resp.get("usage") or {}).get("prompt_tokens", 0),
            "output_tokens": (resp.get("usage") or {}).get("completion_tokens", 0),
        },
    }


def sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def stream_message_start(model: str) -> bytes:
    msg = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    return sse_event("message_start", {"type": "message_start", "message": msg})


def stream_text_block_start(index: int = 0) -> bytes:
    return sse_event("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    })


def stream_text_delta(text: str, index: int = 0) -> bytes:
    return sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def stream_tool_use_block(index: int, tc: dict[str, Any]) -> bytes:
    fn = tc.get("function") or {}
    args = fn.get("arguments") or "{}"
    try:
        input_obj = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        input_obj = {"arguments": args}
    return sse_event("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "name": fn.get("name", ""),
            "input": input_obj or {},
        },
    }) + sse_event("content_block_stop", {"type": "content_block_stop", "index": index})


def stream_content_block_stop(index: int = 0) -> bytes:
    return sse_event("content_block_stop", {"type": "content_block_stop", "index": index})


def stream_message_delta(stop_reason: str) -> bytes:
    return sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })


def stream_message_stop() -> bytes:
    return sse_event("message_stop", {"type": "message_stop"})


def count_tokens_response(req: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(req.get("messages") or [], ensure_ascii=False)
    return {"input_tokens": max(1, len(raw) // 4)}
