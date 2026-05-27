#!/usr/bin/env python3
"""
Qwen Chat → OpenAI-compatible reverse proxy.

Supports session reuse, automatic retry on parse failure, prompt
length monitoring, and structured JSON logging.
Zero dependencies — chỉ dùng Python stdlib.

Usage:
    QWEN_JWT="eyJ..." python3 qwen_proxy/qwen_proxy.py [--port 8080] [--host 127.0.0.1]
"""

import argparse
import hashlib
import http.client
import http.server
import json
import os
import re
import ssl
import sys
import threading
import time
import uuid
from urllib.parse import urlparse

from qwen_proxy import anthropic, toolcall, toolstream
from qwen_proxy.logging_utils import log_event

QWEN_HOST = "chat.qwen.ai"
QWEN_BASE = "/api/v2"
QWEN_TIMEOUT = 180

# Retry/recovery are intentionally off by default. The proxy should translate
# model-emitted tool markup, not invent tool calls from surrounding context.
RETRY_ON_FAIL = os.environ.get("QWEN_RETRY_ON_FAIL", "0").strip() in {"1", "true", "yes"}
TOOL_RECOVERY = os.environ.get("QWEN_TOOL_RECOVERY", "0").strip() in {"1", "true", "yes"}

# Prompt length monitoring / auto-truncation (P2)
# NOTE: Claude Code system prompt + tools = ~57K chars after compaction.
# Only truncate for very long conversations. Qwen handles 100K+ fine.
MAX_PROMPT_CHARS = int(os.environ.get("QWEN_MAX_PROMPT_CHARS", "200000"))

# Session reuse TTL in seconds (P2). 0 = disabled (original behavior).
# Set to e.g. 600 to reuse sessions for 10 minutes.
SESSION_TTL = int(os.environ.get("QWEN_SESSION_TTL", "0"))

# Session cache: model_name -> (chat_id, created_timestamp)
_session_cache: dict[str, tuple[str, float]] = {}
_session_lock = threading.Lock()

MODEL_ALIASES = {
    "qwen3.7-max-preview": "qwen-latest-series-invite-beta-v24",
    "qwen3.7-plus-preview": "qwen-latest-series-invite-beta-v16",
    "qwen3.6-max": "qwen3.6-max-preview",
    "qwen3.5-max-preview": "qwen3.5-max-2026-03-08",
    "qwen3-max": "qwen3-max-2026-01-23",
}

BASE_MODELS = {
    "qwen3.7-max": "qwen3.7-max",
    "qwen3.6-plus": "qwen3.6-plus",
    "qwen3.6-max-preview": "qwen3.6-max-preview",
    "qwen3.6-27b": "qwen3.6-27b",
    "qwen-latest-series-invite-beta-v24": "qwen-latest-series-invite-beta-v24",
    "qwen-latest-series-invite-beta-v16": "qwen-latest-series-invite-beta-v16",
    "qwen3.5-plus": "qwen3.5-plus",
    "qwen3.5-omni-plus": "qwen3.5-omni-plus",
    "qwen3.6-35b-a3b": "qwen3.6-35b-a3b",
    "qwen3.5-flash": "qwen3.5-flash",
    "qwen3.5-max-2026-03-08": "qwen3.5-max-2026-03-08",
    "qwen3.6-plus-preview": "qwen3.6-plus-preview",
    "qwen3.5-397b-a17b": "qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b": "qwen3.5-122b-a10b",
    "qwen3.5-omni-flash": "qwen3.5-omni-flash",
    "qwen3.5-27b": "qwen3.5-27b",
    "qwen3.5-35b-a3b": "qwen3.5-35b-a3b",
    "qwen3-max-2026-01-23": "qwen3-max-2026-01-23",
    "qwen-plus-2025-07-28": "qwen-plus-2025-07-28",
    "qwen3-coder-plus": "qwen3-coder-plus",
    "qwen3-vl-plus": "qwen3-vl-plus",
    "qwen3-omni-flash-2025-12-01": "qwen3-omni-flash-2025-12-01",
    "qwen-max-latest": "qwen-max-latest",
}
BASE_MODELS.update(MODEL_ALIASES)
DEFAULT_MODEL = os.environ.get("QWEN_DEFAULT_MODEL", "qwen3.6-plus")
DEFAULT_TOOL_MODEL = os.environ.get("QWEN_DEFAULT_TOOL_MODEL", "qwen3.6-max-preview").strip()
TOOL_MODEL = os.environ.get("QWEN_TOOL_MODEL", DEFAULT_TOOL_MODEL).strip()

# thinking_mode: "auto" = model decides, "thinking" = always think, "fast" = never think
THINKING_MODES = {"auto", "thinking", "fast"}

MODEL_LIST_BASE_IDS = [
    "qwen3.7-max",
    "qwen3.7-max-preview",
    "qwen3.7-plus-preview",
    "qwen3.6-plus",
    "qwen3.6-max-preview",
    "qwen3.6-27b",
    "qwen3.5-plus",
    "qwen3.5-omni-plus",
    "qwen3.6-35b-a3b",
    "qwen3.5-flash",
    "qwen3.5-max-preview",
    "qwen3.6-plus-preview",
    "qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b",
    "qwen3.5-omni-flash",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3-max",
    "qwen-plus-2025-07-28",
    "qwen3-coder-plus",
    "qwen3-vl-plus",
    "qwen3-omni-flash-2025-12-01",
    "qwen-max-latest",
]

MODELS_LIST = []
for base_id in MODEL_LIST_BASE_IDS:
    for suffix in ("", "-thinking", "-fast"):
        MODELS_LIST.append({"id": f"{base_id}{suffix}", "object": "model", "owned_by": "qwen"})


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate(text: str, limit: int) -> str:
    text = _normalize_ws(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _compact_tool_description(name: str, desc: str) -> str:
    desc = desc or ""
    first_line = desc.strip().splitlines()[0] if desc.strip() else ""

    preferred = {
        "agent": "Launch a specialized sub-agent for a multi-step task.",
        "task_stop": "Stop a running background task by task_id.",
        "send_message": "Send a text message to a running background task.",
        "skill": "Run a named skill before continuing when a relevant skill exists.",
        "list_directory": "List files and directories in an absolute path.",
        "read_file": "Read a file by absolute path, optionally with offset/limit.",
        "grep_search": "Search file contents by regex.",
        "glob": "Find files by glob pattern.",
        "edit": "Edit an existing file.",
        "write_file": "Write content to a file at an absolute path.",
        "run_shell_command": "Run a shell command and return output.",
        "todo_write": "Create or update the task todo list.",
        "ask_user_question": "Ask the user for clarification when necessary.",
        "exit_plan_mode": "Exit planning mode and continue execution.",
        "web_fetch": "Fetch a webpage or URL and return processed content.",
        "Skill": "Load a named skill. Always pass the exact skill id in the `skill` parameter.",
    }
    if name in preferred:
        return preferred[name]
    return _truncate(first_line or desc, 140)


def _compact_param_description(desc: str) -> str:
    return _truncate(desc, 80)


def _compact_system_message(content: str) -> str:
    if content.startswith("You are Qwen Code, an interactive CLI agent"):
        return (
            "You are a CLI coding assistant. Follow the user's request, keep responses concise, "
            "use absolute file paths, and use the provided tools for actions. "
            "For any action involving files, shell, web access, planning, or delegation, "
            "do not describe the action in prose; call the appropriate tool."
        )
    # Long agent system prompts (Claude Code, Cursor, etc.) overwhelm context
    # and cause Qwen to ignore tool-call instructions placed at the top.
    if len(content) > 2000 and _is_agent_system_prompt(content):
        return (
            "You are a CLI coding assistant. Follow the user's request precisely. "
            "Use absolute file paths. For ANY action (reading, writing, editing files, "
            "running commands, searching, browsing), ALWAYS call the appropriate tool. "
            "Never describe what you plan to do — execute it by calling a tool. "
            "Call one tool at a time, then wait for the result before proceeding."
        )
    # Truncate very long system prompts to preserve tool instruction visibility
    if len(content) > 4000:
        return content[:2000].rstrip() + "\n\n(System context truncated for efficiency.)"
    return content


def _is_agent_system_prompt(content: str) -> bool:
    """Detect system prompts from AI coding agents (Claude Code, Cursor, etc.)."""
    if len(content) < 500:
        return False
    head = content[:3000].lower()
    markers = [
        "you are claude", "made by anthropic", "claude code",
        "you are cursor", "you are an ai assistant",
        "tool_use", "function calling", "coding assistant",
    ]
    return sum(1 for m in markers if m in head) >= 2


def _compact_user_context(content: str) -> str:
    if not content.startswith("This is the Qwen Code. We are setting up the context for our chat."):
        return content

    kept = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Today's date is "):
            kept.append(line)
        elif line.startswith("My operating system is:"):
            kept.append(line)
        elif line.startswith("I'm currently working in the directory:"):
            kept.append(line)

    if kept:
        return "Qwen Code session context:\n" + "\n".join(kept)
    return "Qwen Code session context initialized."


def resolve_model(name: str) -> tuple[str, str]:
    """Returns (qwen_model_id, thinking_mode)."""
    name = (name or DEFAULT_MODEL).rsplit("/", 1)[-1]
    if name.endswith("-thinking"):
        base = name[:-9]
        mode = "thinking"
    elif name.endswith("-fast"):
        base = name[:-5]
        mode = "fast"
    else:
        base = name
        mode = "auto"
    qwen_model = BASE_MODELS.get(base, DEFAULT_MODEL)
    return qwen_model, mode


def select_upstream_model(client_model: str, has_tools: bool) -> str:
    if not has_tools:
        return client_model
    if TOOL_MODEL.lower() in {"", "request", "client", "none"}:
        return client_model
    return TOOL_MODEL


def format_tools_prompt(tools: list) -> str:
    lines = []
    exact_names = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        name = fn["name"]
        exact_names.append(name)
        desc = _compact_tool_description(name, fn.get("description", ""))
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = _compact_param_description(pinfo.get("description", ""))
            req = " (required)" if pname in required else ""
            param_parts.append(f"    - {pname}: {ptype}{req}{' — ' + pdesc if pdesc else ''}")
        lines.append(f"- {name}: {desc}")
        if param_parts:
            lines.extend(param_parts)
    if exact_names:
        return "Exact client capability names: " + ", ".join(exact_names) + "\n" + "\n".join(lines)
    return "\n".join(lines)


def _cdata(text: str) -> str:
    return "<![CDATA[" + str(text).replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _format_assistant_tool_calls_for_prompt(tool_calls: list) -> str:
    lines = ["<tool_calls>"]
    for tc in tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"arguments": args}
        if not isinstance(args, dict):
            args = {"arguments": args}
        lines.append(f'  <invoke name="{name}">')
        for key, value in args.items():
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f'    <parameter name="{key}">{_cdata(value)}</parameter>')
        lines.append("  </invoke>")
    lines.append("</tool_calls>")
    return "\n".join(lines)


TOOL_CALL_INSTRUCTION = """You are an action-request serializer for an API client.

When the next assistant step requires a client capability (reading files, running commands, writing files, searching, listing directories, browsing, loading skills, etc.), serialize exactly one request as XML and output nothing else:

<tool_calls>
  <invoke name="CAPABILITY_NAME">
    <parameter name="param"><![CDATA[value]]></parameter>
  </invoke>
</tool_calls>

IMPORTANT:
- You do not execute actions yourself. The API client executes the XML request after your response.
- Never simulate command output, file contents, browser state, skill loading, or any other result.
- The capability names listed below are valid client-side names. Never write prose such as "Tool X does not exist".
- Use only exact names from the available capability list.
- If you request the Skill capability, always include its required `skill` parameter. Example: <parameter name="skill"><![CDATA[pentest-assistant-reasoning]]></parameter>.
- For multi-step tasks, serialize ONE capability request first. You will receive the result, then you can request the next step.
- Put every argument in a <parameter> tag. Use CDATA for paths, commands, file contents, JSON, and multiline values.
- Do not wrap the XML block in Markdown fences.
- Do not use legacy <|tool_call|> tags.
- When no action is needed (greetings, general knowledge), respond normally with text."""


def _tool_choice_instruction(tool_choice) -> str:
    if not tool_choice or tool_choice == "auto":
        return ""
    if tool_choice == "required":
        return "\n\nTool choice: you MUST serialize one available client capability request."
    if tool_choice == "none":
        return "\n\nTool choice: do not serialize a client capability request for this response."
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
        name = fn.get("name")
        if name:
            return f"\n\nTool choice: you MUST serialize a request for the client capability named {name}."
    return ""


def _tool_names_from_tools(tools: list | None) -> list[str]:
    names = []
    for tool in tools or []:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = fn.get("name", "")
        if name:
            names.append(name)
    return names


def _content_to_plain_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if content is None:
        return ""
    return str(content)


def _latest_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _content_to_plain_text(msg.get("content", ""))
    return ""


def _last_tool_result(messages: list) -> tuple[str, str]:
    tool_name_by_id = {}
    last_name = ""
    last_result = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tool_name_by_id[tc.get("id", "")] = fn.get("name", "")
        elif msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            last_name = tool_name_by_id.get(tool_call_id, "")
            last_result = _content_to_plain_text(msg.get("content", ""))
    return last_name, last_result


def _count_tool_calls(messages: list, tool_name: str) -> int:
    count = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            if fn.get("name", "").lower() == tool_name.lower():
                count += 1
    return count


def _find_tool_name(names: list[str], wanted: str) -> str:
    wanted_lower = wanted.lower()
    for name in names:
        if name.lower() == wanted_lower:
            return name
    return ""


def _looks_like_web_search_request(text: str) -> bool:
    lowered = _normalize_ws(text).lower()
    return any(token in lowered for token in (
        "google",
        "web search",
        "search web",
        "tìm kiếm",
        "tìm trên web",
        "tìm bài",
        "vnexpress",
        "mới nhất",
        "latest",
    ))


def _looks_like_empty_search_result(text: str) -> bool:
    lowered = _normalize_ws(text).lower()
    return any(token in lowered for token in (
        "did 0 searches",
        "0 searches",
        "no search results",
        "no results",
        "không có kết quả",
    ))


def _action_hint_for_messages(messages: list, tools: list | None, tool_choice=None) -> str:
    names = _tool_names_from_tools(tools)
    web_search = _find_tool_name(names, "WebSearch")
    if not web_search or tool_choice == "none":
        return ""

    last_tool_name, last_tool_result = _last_tool_result(messages)
    if last_tool_name.lower() == web_search.lower() and _looks_like_empty_search_result(last_tool_result):
        search_count = _count_tool_calls(messages, web_search)
        if search_count < 2:
            return (
                f"The previous {web_search} result had no results. Retry {web_search} once "
                "with a broader query. Do not use other capability names unless they are listed exactly."
            )
        return (
            f"{web_search} already returned no results. Answer honestly that no web results were found. "
            "Do not claim unrelated capabilities do not exist."
        )

    if _looks_like_web_search_request(_latest_user_text(messages)):
        return (
            f"The latest user request is a web search. Use {web_search} first with a concise query "
            "derived from the user text. Do not answer from memory."
        )
    return ""


def flatten_messages(messages: list, tools: list | None = None, tool_choice=None) -> str:
    parts = []
    tool_name_by_id = {}

    if tools:
        tool_desc = format_tools_prompt(tools)
        parts.append(f"[System]\n{TOOL_CALL_INSTRUCTION}\n\nAvailable tools:\n{tool_desc}{_tool_choice_instruction(tool_choice)}")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )

        if role == "system":
            content = _compact_system_message(content)
        elif role == "user":
            content = _compact_user_context(content)

        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    call_id = tc.get("id", "?")
                    tool_name = fn.get("name", "?")
                    tool_name_by_id[call_id] = tool_name
                parts.append(
                    "[Assistant]\nPrevious client capability request already sent:\n"
                    + _format_assistant_tool_calls_for_prompt(tool_calls)
                )
            elif content and _normalize_ws(content).lower() != "got it. thanks for the context!":
                parts.append(f"[Assistant]\n{content}")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "?")
            tool_name = tool_name_by_id.get(tool_call_id, "unknown_tool")
            parts.append(
                f"[Tool Result]\nTool name: {tool_name}\nCall ID: {tool_call_id}\nResult:\n{content}\n"
                "Use this result to continue. If another action is needed, request exactly one next client capability."
            )
        else:
            parts.append(f"[User]\n{content}")

    # Add a soft tool-format reminder at the end of prompt (recency bias).
    # It must not force a new tool call after a tool result; that causes
    # Claude Code-style runners to loop.
    if tools:
        tool_names = _tool_names_from_tools(tools)
        names_str = ", ".join(tool_names[:10]) if tool_names else "the tools above"
        last_role = next((m.get("role", "user") for m in reversed(messages) if isinstance(m, dict)), "")
        if tool_choice == "required" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "function"):
            reminder = (
                f"Your available client capabilities: {names_str}. Tool choice requires an action request. "
                "Emit exactly one XML <tool_calls> block and no prose."
            )
        elif last_role == "tool":
            reminder = (
                f"Your available client capabilities: {names_str}. You just received a tool result. "
                "Use it to answer normally unless another tool is strictly necessary. "
                "If another action is needed, emit exactly one XML <tool_calls> block. "
                "Never write 'Tool X does not exist' in the final answer."
            )
        else:
            reminder = (
                f"Your available client capabilities: {names_str}. If the next step requires an action, "
                "emit exactly one XML <tool_calls> block. If no action is needed, answer normally."
            )
        action_hint = _action_hint_for_messages(messages, tools, tool_choice)
        if action_hint:
            reminder += " " + action_hint
        parts.append(
            "[System Reminder]\n"
            + reminder
        )

    return "\n\n".join(parts)


def qwen_request(method: str, path: str, body: dict | None = None, jwt: str = "") -> http.client.HTTPResponse:
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(QWEN_HOST, context=ctx, timeout=QWEN_TIMEOUT)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jwt}",
    }
    payload = json.dumps(body).encode() if body else None
    conn.request(method, path, body=payload, headers=headers)
    return conn.getresponse()


def create_chat(jwt: str, model: str) -> str:
    resp = qwen_request("POST", f"{QWEN_BASE}/chats/new", {
        "title": "proxy",
        "models": [model],
        "chat_mode": "normal",
        "chat_type": "t2t",
        "timestamp": int(time.time() * 1000),
        "project_id": "",
    }, jwt)
    data = json.loads(resp.read())
    if not data.get("success"):
        raise RuntimeError(f"create_chat failed: {data}")
    return data["data"]["id"]


def get_or_create_chat(jwt: str, model: str) -> str:
    """Return a cached chat_id for the model, or create a new one.

    Sessions are cached per model with a configurable TTL (QWEN_SESSION_TTL).
    When TTL=0, session reuse is disabled (every request creates a new chat).
    """
    if SESSION_TTL <= 0:
        return create_chat(jwt, model)

    now = time.time()
    with _session_lock:
        # Evict expired entries
        expired = [k for k, (_, ts) in _session_cache.items() if now - ts > SESSION_TTL]
        for k in expired:
            del _session_cache[k]

        cached = _session_cache.get(model)
        if cached:
            chat_id, ts = cached
            if now - ts <= SESSION_TTL:
                return chat_id

    # Cache miss or expired — create new
    chat_id = create_chat(jwt, model)
    with _session_lock:
        _session_cache[model] = (chat_id, now)
    return chat_id


def _truncate_conversation(messages: list, tools: list | None, max_chars: int) -> list:
    """Truncate conversation history so the flattened prompt fits max_chars.

    Uses actual flatten_messages() output length for accurate measurement,
    since system messages get heavily compacted during flattening.

    Strategy: progressively remove the oldest non-system messages from the
    middle of the conversation (keeping first 2 + last N).
    """
    if not messages or len(messages) <= 4:
        return messages

    # Check actual flattened size — this accounts for system compaction
    prompt = flatten_messages(messages, tools)
    if len(prompt) <= max_chars:
        return messages

    original_prompt_len = len(prompt)

    # Separate system messages from conversation
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]

    if len(conv_msgs) <= 4:
        return messages  # Too few to truncate meaningfully

    # Keep first 2 conversation messages + progressively fewer trailing messages
    first = conv_msgs[:2]
    remaining = conv_msgs[2:]

    for cut in range(1, len(remaining)):
        candidate = system_msgs + first + remaining[cut:]
        prompt = flatten_messages(candidate, tools)
        if len(prompt) <= max_chars:
            log_event("prompt_truncate",
                       original_messages=len(messages),
                       kept_messages=len(candidate),
                       removed_messages=cut,
                       original_prompt_chars=original_prompt_len,
                       truncated_prompt_chars=len(prompt))
            return candidate

    # Last resort: system + last 2 conversation messages only
    last2 = conv_msgs[-2:]
    candidate = system_msgs + last2
    prompt = flatten_messages(candidate, tools)
    log_event("prompt_truncate",
               original_messages=len(messages),
               kept_messages=len(candidate),
               removed_messages=len(conv_msgs) - 2,
               original_prompt_chars=original_prompt_len,
               truncated_prompt_chars=len(prompt))
    return candidate


def _summarize_tool_calls(calls: list[dict]) -> list[dict]:
    """Extract tool name + arguments from tool calls for logging."""
    out = []
    for tc in calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "?")
        args_str = fn.get("arguments", "{}")
        # Truncate long arguments for logging
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        out.append({"name": name, "arguments": args_str})
    return out


def _tool_choice_requires_action(prompt: str) -> bool:
    return (
        "Tool choice requires an action request" in prompt
        or "Tool choice: you MUST serialize" in prompt
    )


def _looks_like_failed_tool_text(text: str) -> bool:
    lowered = (text or "").lower()
    if "tool " in lowered and ("does not exist" in lowered or "does not exists" in lowered):
        return True
    return any(
        phrase in lowered
        for phrase in ("i'm unable to", "i am unable to", "unable to access", "cannot access")
    )


def _sanitize_assistant_text(text: str, tools: list | None = None) -> str:
    if not text or not tools:
        return text or ""
    cleaned = re.sub(
        r"(?:^|\s+)Tool\s+[A-Za-z0-9_.:/-]+\s+does\s+not\s+exists?\.",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[ \t]{2,}", " ", cleaned).lstrip()


def _sanitize_and_log_assistant_text(source: str, text: str, tools: list | None = None) -> str:
    cleaned = _sanitize_assistant_text(text, tools)
    if cleaned != (text or ""):
        log_event(
            "assistant_text_sanitized",
            source=source,
            original_chars=len(text or ""),
            cleaned_chars=len(cleaned),
            preview=(text or "")[:300],
        )
    return cleaned


def _log_no_tool_response(source: str, prompt: str, text: str, dropped: str = "") -> None:
    if dropped or _tool_choice_requires_action(prompt) or _looks_like_failed_tool_text(text):
        log_event(
            "tool_miss",
            source=source,
            chars=len(text),
            preview=text[:500],
            dropped_markup=dropped[:300] if dropped else None,
        )
    elif text:
        log_event("assistant_text", source=source, chars=len(text), preview=text[:300])
    else:
        log_event("assistant_text", source=source, chars=0, detail="empty response")


def _build_retry_prompt(messages: list, tools: list | None, tool_choice=None) -> str:
    """Build a minimal prompt for retry after SIEVE DROP.

    Uses only: tool instruction + last user message + last tool result.
    This produces a much shorter prompt so Qwen is less likely to
    hallucinate DSML format.
    """
    # Extract last user message and last tool result
    last_user = None
    last_tool = None
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "user" and last_user is None:
            last_user = msg
        elif role == "tool" and last_tool is None:
            last_tool = msg
        if last_user and last_tool:
            break

    retry_msgs = []
    if last_tool:
        retry_msgs.append(last_tool)
    if last_user:
        retry_msgs.append(last_user)
    if not retry_msgs:
        # Fallback: just use the last message
        retry_msgs = messages[-1:] if messages else []

    return flatten_messages(retry_msgs, tools, tool_choice)


def build_qwen_payload(chat_id: str, model: str, prompt: str, thinking_mode: str) -> dict:
    fid = str(uuid.uuid4())
    if thinking_mode == "fast":
        fc = {"thinking_enabled": False, "auto_search": False}
    elif thinking_mode == "thinking":
        fc = {
            "thinking_enabled": True,
            "output_schema": "phase",
            "auto_thinking": False,
            "thinking_mode": "Auto",
            "thinking_format": "summary",
            "auto_search": False,
        }
    else:  # auto
        fc = {
            "thinking_enabled": True,
            "output_schema": "phase",
            "auto_thinking": True,
            "thinking_mode": "Auto",
            "thinking_format": "summary",
            "auto_search": False,
        }
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": None,
        "messages": [{
            "fid": fid,
            "parentId": None,
            "childrenIds": [],
            "role": "user",
            "content": prompt,
            "chat_type": "t2t",
            "feature_config": fc,
            "timestamp": int(time.time()),
            "models": [model],
            "sub_chat_type": "t2t",
            "parent_id": None,
        }],
        "timestamp": int(time.time()),
    }


def make_chunk(completion_id: str, model: str, delta: dict, finish_reason=None) -> str:
    obj = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(obj)}\n\n"


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    jwt: str = ""

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        self._send_json(code, {
            "error": {"message": message, "type": "proxy_error", "code": code}
        })

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/v1/models", "/models"}:
            self._send_json(200, {"object": "list", "data": MODELS_LIST})
        elif anthropic.is_models_path(path):
            self._send_json(200, {
                "data": [{"id": item["id"], "type": "model", "display_name": item["id"]} for item in MODELS_LIST],
                "has_more": False,
                "first_id": MODELS_LIST[0]["id"],
                "last_id": MODELS_LIST[-1]["id"],
            })
        elif path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_error(404, "Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if anthropic.is_count_tokens_path(path):
            try:
                req = json.loads(self._read_body())
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
                return
            self._send_json(200, anthropic.count_tokens_response(req))
            return

        if anthropic.is_messages_path(path):
            self._handle_anthropic_messages()
            return

        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_error(404, "Not found")
            return

        try:
            req = json.loads(self._read_body())
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        messages = req.get("messages", [])
        if not messages:
            self._send_error(400, "messages is required")
            return

        client_model = req.get("model", DEFAULT_MODEL)
        stream = req.get("stream", False)
        tool_choice = req.get("tool_choice")
        tools = req.get("tools") if tool_choice != "none" else None
        has_tools = bool(tools)
        upstream_model_name = select_upstream_model(client_model, has_tools)
        model, thinking_mode = resolve_model(upstream_model_name)
        messages = _truncate_conversation(messages, tools, MAX_PROMPT_CHARS)
        prompt = flatten_messages(messages, tools, tool_choice)
        log_event(
            "prompt_info",
            chars=len(prompt),
            messages=len(messages),
            model=model,
            has_tools=has_tools,
            tools=_tool_names_from_tools(tools)[:12],
            action_hint=_action_hint_for_messages(messages, tools, tool_choice),
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        try:
            chat_id = get_or_create_chat(self.jwt, model)
        except Exception as e:
            self._send_error(502, f"Failed to create Qwen chat: {e}")
            return

        payload = build_qwen_payload(chat_id, model, prompt, thinking_mode)

        try:
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(QWEN_HOST, context=ctx, timeout=QWEN_TIMEOUT)
            conn.request(
                "POST",
                f"{QWEN_BASE}/chat/completions?chat_id={chat_id}",
                body=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.jwt}",
                    "Accept": "application/json",
                },
            )
            upstream = conn.getresponse()
        except Exception as e:
            self._send_error(502, f"Qwen upstream error: {e}")
            return

        if upstream.status != 200:
            body = upstream.read().decode(errors="replace")
            self._send_error(upstream.status, f"Qwen returned {upstream.status}: {body[:500]}")
            return

        if stream:
            self._handle_stream(upstream, completion_id, client_model, thinking_mode, has_tools, tools, prompt)
        else:
            self._handle_non_stream(upstream, completion_id, client_model, thinking_mode, has_tools, tools, prompt)

        try:
            conn.close()
        except Exception:
            pass

    def _collect_upstream(self, upstream) -> tuple[str, int, int]:
        full_content = ""
        input_tokens = 0
        output_tokens = 0
        for raw_line in upstream:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "choices" not in event:
                continue
            delta = event["choices"][0].get("delta", {})
            phase = delta.get("phase", "")
            if phase == "thinking_summary":
                continue
            if phase == "answer" or not phase:
                full_content += delta.get("content", "")
            usage = event.get("usage", {})
            if usage.get("input_tokens"):
                input_tokens = usage["input_tokens"]
            if usage.get("output_tokens"):
                output_tokens = usage["output_tokens"]
        return full_content, input_tokens, output_tokens

    def _write_sse(self, data: bytes):
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _write_stream_tool_calls(self, completion_id: str, model: str, calls: list[dict]):
        for i, tc in enumerate(calls):
            tc_delta = {
                "tool_calls": [{
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }]
            }
            self._write_sse(make_chunk(completion_id, model, tc_delta).encode())

    def _handle_stream(self, upstream, completion_id: str, model: str, thinking_mode: str, has_tools: bool = False, tools: list | None = None, prompt: str = ""):
        if has_tools:
            self._start_sse()
            self._write_sse(make_chunk(completion_id, model, {"role": "assistant"}).encode())
            sieve = toolstream.ToolStreamSieve(tools, context=prompt)
            pending_text = []

            for raw_line in upstream:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if "choices" not in event:
                    continue
                delta = event["choices"][0].get("delta", {})
                phase = delta.get("phase", "")
                if phase == "thinking_summary":
                    continue
                if phase != "answer" and phase:
                    continue

                content = delta.get("content", "")
                if not content:
                    continue
                for evt in sieve.feed(content):
                    if evt.content:
                        pending_text.append(evt.content)
                    if evt.tool_calls:
                        log_event("tool_parse_ok", source="stream", count=len(evt.tool_calls))
                        self._write_stream_tool_calls(completion_id, model, evt.tool_calls)
                        self._write_sse(make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                        self._write_sse(b"data: [DONE]\n\n")
                        return

            for evt in sieve.flush():
                if evt.content:
                    pending_text.append(evt.content)
                if evt.tool_calls:
                    log_event("tool_parse_ok", source="stream_flush", count=len(evt.tool_calls))
                    self._write_stream_tool_calls(completion_id, model, evt.tool_calls)
                    self._write_sse(make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                    self._write_sse(b"data: [DONE]\n\n")
                    return

            # No tool calls detected — try recovery, then retry
            full_text = "".join(pending_text)
            dropped = sieve.buffer or sieve.dropped_markup
            fallback_calls = (
                toolcall.infer_tool_calls_from_context(tools, prompt, full_text or dropped)
                if TOOL_RECOVERY
                else None
            )
            if fallback_calls:
                log_event("tool_recovery", source="stream", count=len(fallback_calls),
                          calls=_summarize_tool_calls(fallback_calls))
                self._write_stream_tool_calls(completion_id, model, fallback_calls)
                self._write_sse(make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                self._write_sse(b"data: [DONE]\n\n")
                return

            # Retry with simplified prompt (P1)
            if RETRY_ON_FAIL and has_tools:
                retry_calls = self._retry_for_tool_calls(tools, prompt, model)
                if retry_calls:
                    log_event("retry_ok", source="stream", count=len(retry_calls))
                    self._write_stream_tool_calls(completion_id, model, retry_calls)
                    self._write_sse(make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                    self._write_sse(b"data: [DONE]\n\n")
                    return

            _log_no_tool_response("stream", prompt, full_text, dropped)
            full_text = _sanitize_and_log_assistant_text("stream", full_text, tools)

            if full_text:
                self._write_sse(make_chunk(completion_id, model, {"content": full_text}).encode())
            self._write_sse(make_chunk(completion_id, model, {}, finish_reason="stop").encode())
            self._write_sse(b"data: [DONE]\n\n")
            return

        self._start_sse()
        self._write_sse(make_chunk(completion_id, model, {"role": "assistant"}).encode())

        for raw_line in upstream:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            if "choices" not in event:
                continue

            delta = event["choices"][0].get("delta", {})
            phase = delta.get("phase", "")

            if phase == "thinking_summary":
                continue

            if phase == "answer" or not phase:
                content = delta.get("content", "")
                if content:
                    self._write_sse(make_chunk(completion_id, model, {"content": content}).encode())

        self._write_sse(make_chunk(completion_id, model, {}, finish_reason="stop").encode())
        self._write_sse(b"data: [DONE]\n\n")

    def _handle_non_stream(self, upstream, completion_id: str, model: str, thinking_mode: str, has_tools: bool = False, tools: list | None = None, prompt: str = ""):
        full_content, input_tokens, output_tokens = self._collect_upstream(upstream)

        if has_tools:
            tool_calls = toolcall.parse_tool_calls(full_content, tools, context=prompt)
            if not tool_calls:
                tool_calls = (
                    toolcall.infer_tool_calls_from_context(tools, prompt, full_content)
                    if TOOL_RECOVERY
                    else None
                )
            if tool_calls:
                self._send_json(200, {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                })
                return

        full_content = _sanitize_and_log_assistant_text("non_stream", full_content, tools if has_tools else None)
        self._send_json(200, {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        })

    def _open_qwen_upstream(self, req: dict, completion_model: str) -> tuple[object | None, str | None, str]:
        tools = req.get("tools") if req.get("tool_choice") != "none" else None
        upstream_model_name = select_upstream_model(req.get("model", DEFAULT_MODEL), bool(tools))
        model, thinking_mode = resolve_model(upstream_model_name)
        messages = _truncate_conversation(req.get("messages", []), tools, MAX_PROMPT_CHARS)
        prompt = flatten_messages(messages, tools, req.get("tool_choice"))
        log_event(
            "prompt_info",
            chars=len(prompt),
            messages=len(messages),
            model=model,
            has_tools=bool(tools),
            tools=_tool_names_from_tools(tools)[:12],
            action_hint=_action_hint_for_messages(messages, tools, req.get("tool_choice")),
        )
        try:
            chat_id = get_or_create_chat(self.jwt, model)
        except Exception as e:
            return None, f"Failed to create Qwen chat: {e}", prompt
        payload = build_qwen_payload(chat_id, model, prompt, thinking_mode)
        try:
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(QWEN_HOST, context=ctx, timeout=QWEN_TIMEOUT)
            conn.request(
                "POST",
                f"{QWEN_BASE}/chat/completions?chat_id={chat_id}",
                body=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.jwt}",
                    "Accept": "application/json",
                },
            )
            upstream = conn.getresponse()
        except Exception as e:
            return None, f"Qwen upstream error: {e}", prompt
        if upstream.status != 200:
            body = upstream.read().decode(errors="replace")
            return None, f"Qwen returned {upstream.status}: {body[:500]}", prompt
        return upstream, None, prompt

    def _open_qwen_upstream_simple(self, prompt: str, model: str) -> tuple[object | None, str | None]:
        """Open a Qwen upstream connection with a pre-built prompt (for retry)."""
        _, thinking_mode = resolve_model(model)
        try:
            chat_id = create_chat(self.jwt, resolve_model(model)[0])
        except Exception as e:
            return None, f"Retry create_chat failed: {e}"
        payload = build_qwen_payload(chat_id, resolve_model(model)[0], prompt, thinking_mode)
        try:
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(QWEN_HOST, context=ctx, timeout=QWEN_TIMEOUT)
            conn.request(
                "POST",
                f"{QWEN_BASE}/chat/completions?chat_id={chat_id}",
                body=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.jwt}",
                    "Accept": "application/json",
                },
            )
            upstream = conn.getresponse()
        except Exception as e:
            return None, f"Retry upstream error: {e}"
        if upstream.status != 200:
            body = upstream.read().decode(errors="replace")
            return None, f"Retry Qwen returned {upstream.status}: {body[:500]}"
        return upstream, None

    def _retry_for_tool_calls(
        self,
        tools: list | None,
        original_prompt: str,
        model: str,
    ) -> list[dict] | None:
        """Retry with a simplified prompt after tool parse failure.

        Returns parsed tool_calls on success, None on failure.
        """
        if not tools:
            return None

        # Build a much shorter prompt with just tool instruction + key context
        retry_prompt = (
            TOOL_CALL_INSTRUCTION
            + "\n\nAvailable tools:\n"
            + format_tools_prompt(tools)
            + "\n\n[System Reminder]\nThe previous response had a formatting error. "
            "Please respond with EXACTLY one XML <tool_calls> block. "
            "Do NOT write any text before or after the block.\n\n"
        )
        # Append the tail of the original prompt (last ~2000 chars) for context
        if len(original_prompt) > 2000:
            retry_prompt += "[Context (last part)]\n" + original_prompt[-2000:]
        else:
            retry_prompt += original_prompt

        log_event("retry_attempt", prompt_chars=len(retry_prompt), model=model)

        upstream, err = self._open_qwen_upstream_simple(retry_prompt, model)
        if err:
            log_event("retry_fail", error=err)
            return None

        full_content, _, _ = self._collect_upstream(upstream)
        try:
            upstream.close()
        except Exception:
            pass

        if not full_content:
            log_event("retry_fail", detail="empty response")
            return None

        calls = toolcall.parse_tool_calls(full_content, tools, context=original_prompt)
        if calls:
            return calls

        # Try inference as last resort on retry text too
        calls = (
            toolcall.infer_tool_calls_from_context(tools, original_prompt, full_content)
            if TOOL_RECOVERY
            else None
        )
        if calls:
            return calls

        log_event("retry_fail", detail="parse failed on retry", chars=len(full_content),
                  preview=full_content[:300])
        return None

    def _handle_anthropic_messages(self):
        try:
            anthropic_req = json.loads(self._read_body())
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        openai_req = anthropic.to_openai_request(anthropic_req, DEFAULT_MODEL)
        model = anthropic_req.get("model") or openai_req.get("model", DEFAULT_MODEL)
        tools = openai_req.get("tools")
        stream = bool(openai_req.get("stream"))
        upstream, err, prompt = self._open_qwen_upstream(openai_req, model)
        if err:
            self._send_error(502, err)
            return

        if stream:
            self._handle_anthropic_stream(upstream, model, bool(tools), tools, prompt)
            return

        full_content, input_tokens, output_tokens = self._collect_upstream(upstream)
        tool_calls = toolcall.parse_tool_calls(full_content, tools, context=prompt) if tools else None
        if tools and not tool_calls:
            tool_calls = (
                toolcall.infer_tool_calls_from_context(tools, prompt, full_content)
                if TOOL_RECOVERY
                else None
            )
        if not tool_calls:
            full_content = _sanitize_and_log_assistant_text("anthropic_non_stream", full_content, tools)
        openai_resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None if tool_calls else full_content, "tool_calls": tool_calls or []},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        if not tool_calls:
            openai_resp["choices"][0]["message"].pop("tool_calls", None)
        self._send_json(200, anthropic.from_openai_response(openai_resp, model))

    def _handle_anthropic_stream(self, upstream, model: str, has_tools: bool, tools: list | None, prompt: str = ""):
        self._start_sse()
        self._write_sse(anthropic.stream_message_start(model))
        text_started = False
        block_index = 0
        sieve = toolstream.ToolStreamSieve(tools, context=prompt) if has_tools else None
        streamed_text: list[str] = []

        for raw_line in upstream:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "choices" not in event:
                continue
            delta = event["choices"][0].get("delta", {})
            phase = delta.get("phase", "")
            if phase == "thinking_summary":
                continue
            if phase != "answer" and phase:
                continue
            content = delta.get("content", "")
            if not content:
                continue

            events = sieve.feed(content) if sieve else [toolstream.ToolStreamEvent(content=content)]
            for evt in events:
                if evt.content:
                    streamed_text.append(evt.content)
                    if not has_tools:
                        if not text_started:
                            self._write_sse(anthropic.stream_text_block_start(block_index))
                            text_started = True
                        self._write_sse(anthropic.stream_text_delta(evt.content, block_index))
                if evt.tool_calls:
                    log_event("tool_parse_ok", source="anthropic_stream", count=len(evt.tool_calls))
                    if text_started:
                        self._write_sse(anthropic.stream_content_block_stop(block_index))
                        block_index += 1
                        text_started = False
                    for tc in evt.tool_calls:
                        self._write_sse(anthropic.stream_tool_use_block(block_index, tc))
                        block_index += 1
                    self._write_sse(anthropic.stream_message_delta("tool_use"))
                    self._write_sse(anthropic.stream_message_stop())
                    return

        if sieve:
            for evt in sieve.flush():
                if evt.content:
                    streamed_text.append(evt.content)
                    if not has_tools:
                        if not text_started:
                            self._write_sse(anthropic.stream_text_block_start(block_index))
                            text_started = True
                        self._write_sse(anthropic.stream_text_delta(evt.content, block_index))
                if evt.tool_calls:
                    log_event("tool_parse_ok", source="anthropic_stream_flush", count=len(evt.tool_calls))
                    if text_started:
                        self._write_sse(anthropic.stream_content_block_stop(block_index))
                        block_index += 1
                        text_started = False
                    for tc in evt.tool_calls:
                        self._write_sse(anthropic.stream_tool_use_block(block_index, tc))
                        block_index += 1
                    self._write_sse(anthropic.stream_message_delta("tool_use"))
                    self._write_sse(anthropic.stream_message_stop())
                    return

        # No tool calls detected in Anthropic stream — try recovery, then retry
        if has_tools:
            full_text = "".join(streamed_text)
            dropped = sieve.dropped_markup if sieve else ""
            fallback_calls = (
                toolcall.infer_tool_calls_from_context(tools, prompt, full_text or dropped)
                if TOOL_RECOVERY
                else None
            )
            if fallback_calls:
                log_event("tool_recovery", source="anthropic_stream", count=len(fallback_calls),
                          calls=_summarize_tool_calls(fallback_calls),
                          model_text=full_text[:300] if full_text else None)
                if text_started:
                    self._write_sse(anthropic.stream_content_block_stop(block_index))
                    block_index += 1
                    text_started = False
                for tc in fallback_calls:
                    self._write_sse(anthropic.stream_tool_use_block(block_index, tc))
                    block_index += 1
                self._write_sse(anthropic.stream_message_delta("tool_use"))
                self._write_sse(anthropic.stream_message_stop())
                return

            # Retry with simplified prompt (P1)
            if RETRY_ON_FAIL:
                retry_calls = self._retry_for_tool_calls(tools, prompt, model)
                if retry_calls:
                    log_event("retry_ok", source="anthropic_stream", count=len(retry_calls))
                    if text_started:
                        self._write_sse(anthropic.stream_content_block_stop(block_index))
                        block_index += 1
                        text_started = False
                    for tc in retry_calls:
                        self._write_sse(anthropic.stream_tool_use_block(block_index, tc))
                        block_index += 1
                    self._write_sse(anthropic.stream_message_delta("tool_use"))
                    self._write_sse(anthropic.stream_message_stop())
                    return

            _log_no_tool_response("anthropic_stream", prompt, full_text, dropped)
            full_text = _sanitize_and_log_assistant_text("anthropic_stream", full_text, tools)
            if full_text and not text_started:
                self._write_sse(anthropic.stream_text_block_start(block_index))
                text_started = True
                self._write_sse(anthropic.stream_text_delta(full_text, block_index))

        if text_started:
            self._write_sse(anthropic.stream_content_block_stop(block_index))
        self._write_sse(anthropic.stream_message_delta("end_turn"))
        self._write_sse(anthropic.stream_message_stop())


def main():
    parser = argparse.ArgumentParser(description="Qwen → OpenAI proxy")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--jwt", default=os.environ.get("QWEN_JWT", ""))
    args = parser.parse_args()

    if not args.jwt:
        print("Error: JWT token required. Set QWEN_JWT env var or use --jwt", file=sys.stderr)
        sys.exit(1)

    ProxyHandler.jwt = args.jwt

    server = http.server.ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(f"Qwen proxy listening on http://{args.host}:{args.port}")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/messages")
    print(f"  POST /anthropic/v1/messages")
    print(f"  GET  /v1/models")
    print(f"  GET  /anthropic/v1/models")
    print(f"  GET  /health")
    print(f"  Models: {', '.join(MODEL_LIST_BASE_IDS)}")
    print(f"  Tool upstream model: {TOOL_MODEL or 'request model'}")
    print(f"  Suffix: (none)=auto, -thinking=always think, -fast=no think")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
