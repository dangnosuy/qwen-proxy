#!/usr/bin/env python3
"""Qwen Proxy with optional raw model mode + auto-retry on empty response.

Khác biệt so với qwen_proxy.py gốc:
  - Non-tool requests use the client model directly
  - Tool requests default to the stable tool model, matching server.py
  - Set QWEN_RAW_TOOL_MODEL=none to force client model for tool requests
  - Auto-retry lên đến MAX_EMPTY_RETRIES lần khi upstream trả empty
  - Vẫn giữ nguyên tool call parsing (XML → OpenAI format)
  - Vẫn giữ Anthropic compat, streaming, etc.

Usage:
    QWEN_JWT="eyJ..." python3 qwen_proxy_raw.py [--port 8080] [--host 127.0.0.1]

Env vars:
    QWEN_RAW_MAX_RETRIES  — max empty-response retries (default: 2)
    QWEN_RAW_TOOL_MODEL   — tool upstream model, default qwen3.6-max-preview; set none for true raw
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import http.client
import http.server
import json
import os
import ssl
import time
import uuid

from qwen_proxy import anthropic, toolcall, toolstream
from qwen_proxy.logging_utils import log_event
import qwen_proxy.server as _srv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_EMPTY_RETRIES = int(os.environ.get("QWEN_RAW_MAX_RETRIES", "2"))
RAW_TOOL_MODEL = os.environ.get("QWEN_RAW_TOOL_MODEL", _srv.DEFAULT_TOOL_MODEL).strip()


def _raw_select_upstream_model(client_model: str, has_tools: bool) -> str:
    """Raw mode for normal chat; stable model fallback for tool calls."""
    if not has_tools:
        return client_model
    if RAW_TOOL_MODEL.lower() in {"", "request", "client", "none"}:
        return client_model
    return RAW_TOOL_MODEL


# Monkey-patch: keep raw normal chat, but preserve stable tool fallback by default.
_srv.select_upstream_model = _raw_select_upstream_model
_srv.TOOL_MODEL = RAW_TOOL_MODEL

# Enable tool recovery (infer tool calls from context when model fails)
_srv.TOOL_RECOVERY = True


def _should_retry(answer: str, has_tools: bool) -> bool:
    """Check if the response should trigger a retry.

    Only empty upstream responses are retried. Tool refusal text is handled by
    parser recovery/sanitization so a bad tool answer does not multiply latency.
    """
    text = (answer or "").strip()
    if not text:
        return True  # Empty response
    return False


# ---------------------------------------------------------------------------
# RawProxyHandler — extends ProxyHandler with auto-retry on empty
# ---------------------------------------------------------------------------
class RawProxyHandler(_srv.ProxyHandler):
    """Proxy handler that retries on empty upstream responses.

    qwen3.7-max intermittently returns empty streams (0 bytes, ~1s).
    This handler detects that and retries automatically.
    """

    # ------------------------------------------------------------------
    # Helper: open a fresh upstream connection + get response
    # ------------------------------------------------------------------
    def _open_fresh_upstream(self, model: str, prompt: str, thinking_mode: str):
        """Create a new chat + send prompt, return (upstream_response, conn) or raise."""
        qwen_model, _ = _srv.resolve_model(model)
        chat_id = _srv.create_chat(self.jwt, qwen_model)
        payload = _srv.build_qwen_payload(chat_id, qwen_model, prompt, thinking_mode)

        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(_srv.QWEN_HOST, context=ctx, timeout=_srv.QWEN_TIMEOUT)
        conn.request(
            "POST",
            f"{_srv.QWEN_BASE}/chat/completions?chat_id={chat_id}",
            body=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.jwt}",
                "Accept": "application/json",
            },
        )
        upstream = conn.getresponse()
        return upstream, conn

    # ------------------------------------------------------------------
    # Helper: consume upstream and check if it's empty
    # ------------------------------------------------------------------
    def _collect_upstream_full(self, upstream) -> tuple[str, str, int, int, list]:
        """Collect ALL SSE events from upstream, return (answer, thinking, in_tok, out_tok, raw_lines)."""
        answer = ""
        thinking = ""
        input_tokens = 0
        output_tokens = 0
        raw_lines = []

        for raw_line in upstream:
            line = raw_line.decode("utf-8", errors="replace").strip()
            raw_lines.append(line)
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
                extra = delta.get("extra", {})
                thought = extra.get("summary_thought", {}).get("content", [])
                if thought:
                    thinking += " ".join(thought)
                continue
            if phase == "answer" or not phase:
                answer += delta.get("content", "")
            usage = event.get("usage", {})
            if usage.get("input_tokens"):
                input_tokens = usage["input_tokens"]
            if usage.get("output_tokens"):
                output_tokens = usage["output_tokens"]

        return answer, thinking, input_tokens, output_tokens, raw_lines

    # ------------------------------------------------------------------
    # Shared retry logic
    # ------------------------------------------------------------------
    def _retry_upstream(self, answer: str, has_tools: bool, model: str,
                        prompt: str, source: str) -> tuple[str, str, int, int]:
        """Retry upstream if answer should be retried. Returns (answer, thinking, in_tok, out_tok)."""
        retries = 0
        thinking = ""
        in_tok = out_tok = 0
        while _should_retry(answer, has_tools) and retries < MAX_EMPTY_RETRIES:
            retries += 1
            reason = "empty"
            retry_model = _srv.select_upstream_model(model, has_tools)
            log_event("raw_retry", attempt=retries, model=model, reason=reason,
                      source=source, upstream_model=retry_model,
                      preview=answer[:100] if answer else "(empty)")
            try:
                _, thinking_mode = _srv.resolve_model(retry_model)
                new_upstream, new_conn = self._open_fresh_upstream(retry_model, prompt, thinking_mode)
                if new_upstream.status != 200:
                    body = new_upstream.read().decode(errors="replace")
                    log_event("raw_retry_fail", attempt=retries, status=new_upstream.status,
                              detail=body[:300])
                    try:
                        new_conn.close()
                    except Exception:
                        pass
                    break
                answer, thinking, in_tok, out_tok, _ = self._collect_upstream_full(new_upstream)
                try:
                    new_conn.close()
                except Exception:
                    pass
            except Exception as e:
                log_event("raw_retry_error", attempt=retries, error=str(e))
                break

        if retries > 0:
            ok = not _should_retry(answer, has_tools)
            log_event("raw_retry_done", attempts=retries, success=ok,
                      answer_chars=len(answer), source=source)

        return answer, thinking, in_tok, out_tok

    def _try_tool_recovery(self, answer: str, has_tools: bool,
                           tools: list | None, prompt: str,
                           source: str) -> list | None:
        """Try to infer tool calls from context when model fails to emit them."""
        if not has_tools or not tools:
            return None
        recovery = toolcall.infer_tool_calls_from_context(tools, prompt, answer)
        if recovery:
            log_event("tool_recovery", source=source, count=len(recovery),
                      calls=_srv._summarize_tool_calls(recovery),
                      model_text=answer[:200] if answer else "(empty)")
        return recovery

    # ------------------------------------------------------------------
    # Override: Anthropic streaming with retry + recovery
    # ------------------------------------------------------------------
    def _handle_anthropic_stream(self, upstream, model: str, has_tools: bool,
                                  tools: list | None, prompt: str = ""):
        """Anthropic SSE stream with auto-retry and tool recovery."""

        if not self._start_sse():
            return
        if not self._write_sse(anthropic.stream_message_start(model)):
            return

        # ---- Collect upstream fully (buffer) so we can retry if needed ----
        answer, thinking, in_tok, out_tok, raw_lines = self._collect_upstream_full(upstream)

        # ---- Retry on empty upstream response only ----
        answer, thinking, in_tok, out_tok = self._retry_upstream(
            answer, has_tools, model, prompt, "anthropic_stream")

        # ---- Now stream the collected response to client ----
        text_started = False
        block_index = 0

        # Try to parse tool calls from the collected answer
        if has_tools and answer.strip():
            tool_calls = toolcall.parse_tool_calls(answer, tools, context=prompt)
            if not tool_calls:
                # Fallback: infer tool calls from context
                tool_calls = self._try_tool_recovery(
                    answer, has_tools, tools, prompt, "raw_anthropic")
            if tool_calls:
                log_event("tool_parse_ok", source="raw_anthropic", count=len(tool_calls))
                for tc in tool_calls:
                    self._write_sse(anthropic.stream_tool_use_block(block_index, tc))
                    block_index += 1
                self._write_sse(anthropic.stream_message_delta("tool_use"))
                self._write_sse(anthropic.stream_message_stop())
                return

        # No tool calls — try recovery on empty too
        if has_tools and not answer.strip():
            tool_calls = self._try_tool_recovery(
                answer, has_tools, tools, prompt, "raw_anthropic_empty")
            if tool_calls:
                log_event("tool_recovery_ok", source="raw_anthropic_empty", count=len(tool_calls))
                for tc in tool_calls:
                    self._write_sse(anthropic.stream_tool_use_block(block_index, tc))
                    block_index += 1
                self._write_sse(anthropic.stream_message_delta("tool_use"))
                self._write_sse(anthropic.stream_message_stop())
                return

        # Send as text (after sanitizing)
        if answer.strip():
            if has_tools:
                answer = _srv._sanitize_and_log_assistant_text("raw_anthropic", answer, tools)
            if answer:
                self._write_sse(anthropic.stream_text_block_start(block_index))
                text_started = True
                self._write_sse(anthropic.stream_text_delta(answer, block_index))

        if text_started:
            self._write_sse(anthropic.stream_content_block_stop(block_index))
        self._write_sse(anthropic.stream_message_delta("end_turn"))
        self._write_sse(anthropic.stream_message_stop())

    # ------------------------------------------------------------------
    # Override: OpenAI streaming with retry + recovery
    # ------------------------------------------------------------------
    def _handle_stream(self, upstream, completion_id: str, model: str,
                       thinking_mode: str, has_tools: bool = False,
                       tools: list | None = None, prompt: str = ""):
        """OpenAI SSE stream with auto-retry and tool recovery."""

        if not self._start_sse():
            return
        if not self._write_sse(_srv.make_chunk(completion_id, model, {"role": "assistant"}).encode()):
            return

        answer, thinking, in_tok, out_tok, raw_lines = self._collect_upstream_full(upstream)

        # Retry on empty upstream response only
        answer, thinking, in_tok, out_tok = self._retry_upstream(
            answer, has_tools, model, prompt, "openai_stream")

        # Stream collected response

        if has_tools and answer.strip():
            tool_calls = toolcall.parse_tool_calls(answer, tools, context=prompt)
            if not tool_calls:
                tool_calls = self._try_tool_recovery(
                    answer, has_tools, tools, prompt, "raw_openai")
            if tool_calls:
                log_event("tool_parse_ok", source="raw_openai", count=len(tool_calls))
                self._write_stream_tool_calls(completion_id, model, tool_calls)
                self._write_sse(_srv.make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                self._write_sse(b"data: [DONE]\n\n")
                return

        # Try recovery on empty
        if has_tools and not answer.strip():
            tool_calls = self._try_tool_recovery(
                answer, has_tools, tools, prompt, "raw_openai_empty")
            if tool_calls:
                self._write_stream_tool_calls(completion_id, model, tool_calls)
                self._write_sse(_srv.make_chunk(completion_id, model, {}, finish_reason="tool_calls").encode())
                self._write_sse(b"data: [DONE]\n\n")
                return

        # Text response
        if answer.strip():
            if has_tools:
                answer = _srv._sanitize_and_log_assistant_text("raw_openai", answer, tools)
            if answer:
                self._write_sse(_srv.make_chunk(completion_id, model, {"content": answer}).encode())

        self._write_sse(_srv.make_chunk(completion_id, model, {}, finish_reason="stop").encode())
        self._write_sse(b"data: [DONE]\n\n")

    # ------------------------------------------------------------------
    # Override: Non-streaming with retry + recovery
    # ------------------------------------------------------------------
    def _handle_non_stream(self, upstream, completion_id: str, model: str,
                           thinking_mode: str, has_tools: bool = False,
                           tools: list | None = None, prompt: str = ""):
        """Non-streaming response with auto-retry and tool recovery."""
        answer, thinking, in_tok, out_tok, raw_lines = self._collect_upstream_full(upstream)

        # Retry on empty upstream response only
        answer, thinking, in_tok, out_tok = self._retry_upstream(
            answer, has_tools, model, prompt, "non_stream")

        # Parse tool calls
        tool_calls = None
        if has_tools and answer.strip():
            tool_calls = toolcall.parse_tool_calls(answer, tools, context=prompt)
            if not tool_calls:
                tool_calls = self._try_tool_recovery(
                    answer, has_tools, tools, prompt, "raw_non_stream")

        # Recovery on empty
        if has_tools and not tool_calls and not answer.strip():
            tool_calls = self._try_tool_recovery(
                answer, has_tools, tools, prompt, "raw_non_stream_empty")

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
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                },
            })
        else:
            if has_tools:
                answer = _srv._sanitize_and_log_assistant_text("raw_non_stream", answer, tools)
            self._send_json(200, {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                },
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen -> OpenAI raw-chat proxy with stable tool fallback")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--jwt", default=os.environ.get("QWEN_JWT", ""))
    args = parser.parse_args()

    if not args.jwt:
        print("Error: JWT token required. Set QWEN_JWT env var or use --jwt", file=sys.stderr)
        sys.exit(1)

    RawProxyHandler.jwt = args.jwt

    server = http.server.ThreadingHTTPServer((args.host, args.port), RawProxyHandler)
    print(f"Qwen proxy listening on http://{args.host}:{args.port}")
    print(f"   Mode: raw normal chat + stable tool fallback")
    print(f"   Empty retries: {MAX_EMPTY_RETRIES}")
    print(f"   POST /v1/chat/completions")
    print(f"   POST /v1/messages")
    print(f"   POST /anthropic/v1/messages")
    print(f"   GET  /v1/models")
    print(f"   GET  /health")
    print(f"   Default model: {_srv.DEFAULT_MODEL}")
    print(f"   Tool upstream model: {RAW_TOOL_MODEL or 'request model'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
