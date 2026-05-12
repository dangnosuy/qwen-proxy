#!/usr/bin/env python3
"""
Qwen Chat → OpenAI-compatible reverse proxy.

Mỗi request tạo chat session mới (stateless).
Zero dependencies — chỉ dùng Python stdlib.

Usage:
    QWEN_JWT="eyJ..." python3 qwen_proxy/qwen_proxy.py [--port 8080] [--host 127.0.0.1]
"""

import argparse
import http.client
import http.server
import json
import os
import re
import ssl
import sys
import time
import uuid
from urllib.parse import urlparse

from qwen_proxy import anthropic, toolcall, toolstream

QWEN_HOST = "chat.qwen.ai"
QWEN_BASE = "/api/v2"
QWEN_TIMEOUT = 180

BASE_MODELS = {
    "qwen3.6-plus": "qwen3.6-plus",
    "qwen3.6-max": "qwen3.6-max-preview",
    "qwen3.6-max-preview": "qwen3.6-max-preview",
    "qwen3.6-27b": "qwen3.6-27b",
}
DEFAULT_MODEL = "qwen3.6-plus"

# thinking_mode: "auto" = model decides, "thinking" = always think, "fast" = never think
THINKING_MODES = {"auto", "thinking", "fast"}

MODELS_LIST = []
for base_id in ["qwen3.6-plus", "qwen3.6-max-preview", "qwen3.6-27b"]:
    MODELS_LIST.append({"id": base_id, "object": "model", "owned_by": "qwen"})
    MODELS_LIST.append({"id": f"{base_id}-thinking", "object": "model", "owned_by": "qwen"})


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
    return content


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
        return "Exact tool names you may call: " + ", ".join(exact_names) + "\n" + "\n".join(lines)
    return "\n".join(lines)


def _cdata(text: str) -> str:
    return "<![CDATA[" + str(text).replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _format_assistant_tool_calls_for_prompt(tool_calls: list) -> str:
    lines = ["<|DSML|tool_calls>"]
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
        lines.append(f'  <|DSML|invoke name="{name}">')
        for key, value in args.items():
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f'    <|DSML|parameter name="{key}">{_cdata(value)}</|DSML|parameter>')
        lines.append("  </|DSML|invoke>")
    lines.append("</|DSML|tool_calls>")
    return "\n".join(lines)


TOOL_CALL_INSTRUCTION = """You are a function-calling AI. You have access to the tools listed below.

When the user's request requires an action (reading files, running commands, writing files, searching, listing directories, etc.), respond with EXACTLY one DSML tool-call block and NOTHING else:

<|DSML|tool_calls>
  <|DSML|invoke name="TOOL_NAME">
    <|DSML|parameter name="param"><![CDATA[value]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>

IMPORTANT:
- You CANNOT perform actions yourself. You MUST use tools. Never simulate or fake results.
- Every tool listed below is available and working. Never say a tool "does not exist".
- Use only exact tool names from the available tools list.
- The API client will execute the tool after you return a tool call. Do not say you have already opened, read, clicked, searched, or written anything before receiving a tool result.
- For multi-step tasks, call ONE tool first. You will receive the result, then you can call the next tool.
- Put every argument in a <|DSML|parameter> tag. Use CDATA for paths, commands, file contents, JSON, and multiline values.
- Do not wrap the DSML block in Markdown fences.
- Do not use legacy <|tool_call|> tags.
- When no action is needed (greetings, general knowledge), respond normally with text."""


def _tool_choice_instruction(tool_choice) -> str:
    if not tool_choice or tool_choice == "auto":
        return ""
    if tool_choice == "required":
        return "\n\nTool choice: you MUST call one of the available tools."
    if tool_choice == "none":
        return "\n\nTool choice: do not call tools for this request."
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
        name = fn.get("name")
        if name:
            return f"\n\nTool choice: you MUST call the tool named {name}."
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
                    "[Assistant]\nPrevious tool call request already sent to the client:\n"
                    + _format_assistant_tool_calls_for_prompt(tool_calls)
                )
            elif content and _normalize_ws(content).lower() != "got it. thanks for the context!":
                parts.append(f"[Assistant]\n{content}")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "?")
            tool_name = tool_name_by_id.get(tool_call_id, "unknown_tool")
            parts.append(
                f"[Tool Result]\nTool name: {tool_name}\nCall ID: {tool_call_id}\nResult:\n{content}\n"
                "Use this result to continue. If another action is needed, call exactly one next tool."
            )
        else:
            parts.append(f"[User]\n{content}")

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

        model, thinking_mode = resolve_model(req.get("model", DEFAULT_MODEL))
        client_model = req.get("model", DEFAULT_MODEL)
        stream = req.get("stream", False)
        tool_choice = req.get("tool_choice")
        tools = req.get("tools") if tool_choice != "none" else None
        prompt = flatten_messages(messages, tools, tool_choice)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        try:
            chat_id = create_chat(self.jwt, model)
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

        has_tools = bool(tools)
        if stream:
            self._handle_stream(upstream, completion_id, client_model, thinking_mode, has_tools, tools)
        else:
            self._handle_non_stream(upstream, completion_id, client_model, thinking_mode, has_tools, tools)

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

    def _handle_stream(self, upstream, completion_id: str, model: str, thinking_mode: str, has_tools: bool = False, tools: list | None = None):
        if has_tools:
            self._start_sse()
            self._write_sse(make_chunk(completion_id, model, {"role": "assistant"}).encode())
            sieve = toolstream.ToolStreamSieve(tools)

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
                        self._write_sse(make_chunk(completion_id, model, {"content": evt.content}).encode())
                    if evt.tool_calls:
                        self.log_message("stream tools OK: %d calls", len(evt.tool_calls))
                        self._write_stream_tool_calls(completion_id, model, evt.tool_calls)
                        self._write_sse(make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                        self._write_sse(b"data: [DONE]\n\n")
                        return

            for evt in sieve.flush():
                if evt.content:
                    self._write_sse(make_chunk(completion_id, model, {"content": evt.content}).encode())
                if evt.tool_calls:
                    self.log_message("stream tools OK: %d calls", len(evt.tool_calls))
                    self._write_stream_tool_calls(completion_id, model, evt.tool_calls)
                    self._write_sse(make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                    self._write_sse(b"data: [DONE]\n\n")
                    return

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

    def _handle_non_stream(self, upstream, completion_id: str, model: str, thinking_mode: str, has_tools: bool = False, tools: list | None = None):
        full_content, input_tokens, output_tokens = self._collect_upstream(upstream)

        if has_tools:
            tool_calls = toolcall.parse_tool_calls(full_content, tools)
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

    def _open_qwen_upstream(self, req: dict, completion_model: str) -> tuple[object | None, str | None]:
        model, thinking_mode = resolve_model(req.get("model", DEFAULT_MODEL))
        tools = req.get("tools") if req.get("tool_choice") != "none" else None
        prompt = flatten_messages(req.get("messages", []), tools, req.get("tool_choice"))
        try:
            chat_id = create_chat(self.jwt, model)
        except Exception as e:
            return None, f"Failed to create Qwen chat: {e}"
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
            return None, f"Qwen upstream error: {e}"
        if upstream.status != 200:
            body = upstream.read().decode(errors="replace")
            return None, f"Qwen returned {upstream.status}: {body[:500]}"
        return upstream, None

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
        upstream, err = self._open_qwen_upstream(openai_req, model)
        if err:
            self._send_error(502, err)
            return

        if stream:
            self._handle_anthropic_stream(upstream, model, bool(tools), tools)
            return

        full_content, input_tokens, output_tokens = self._collect_upstream(upstream)
        tool_calls = toolcall.parse_tool_calls(full_content, tools) if tools else None
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

    def _handle_anthropic_stream(self, upstream, model: str, has_tools: bool, tools: list | None):
        self._start_sse()
        self._write_sse(anthropic.stream_message_start(model))
        text_started = False
        block_index = 0
        sieve = toolstream.ToolStreamSieve(tools) if has_tools else None

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
                    if not text_started:
                        self._write_sse(anthropic.stream_text_block_start(block_index))
                        text_started = True
                    self._write_sse(anthropic.stream_text_delta(evt.content, block_index))
                if evt.tool_calls:
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
                    if not text_started:
                        self._write_sse(anthropic.stream_text_block_start(block_index))
                        text_started = True
                    self._write_sse(anthropic.stream_text_delta(evt.content, block_index))
                if evt.tool_calls:
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
    print(f"  Models: qwen3.6-plus, qwen3.6-max-preview, qwen3.6-27b")
    print(f"  Suffix: (none)=auto, -thinking=always think, -fast=no think")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
