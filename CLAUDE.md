# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`qwen_proxy` is a reverse proxy that turns the **Qwen Chat website** (`chat.qwen.ai`, the consumer web UI behind `/api/v2`) into an **OpenAI- and Anthropic-compatible API**. It speaks `/v1/chat/completions`, `/v1/messages`, and `/anthropic/v1/messages` so agent runners (Claude Code, Cursor, aider, LiteLLM, …) can drive Qwen models. Zero dependencies — Python ≥ 3.11 stdlib only.

> This is a standalone project. The parent `../CLAUDE.md` describes an unrelated security-assessment platform; ignore it when working here. (The pentest-specific recovery heuristics in `toolcall.py` exist only because this proxy is used to drive those agent runners — see Gotchas.)

## Commands

```bash
# Run the default server (stable-tool-fallback mode). JWT is required.
QWEN_JWT="eyJ..." python3 qwen_proxy.py --port 8080 --host 127.0.0.1

# Run raw mode (buffered stream + empty-response auto-retry + tool recovery)
QWEN_JWT="eyJ..." python3 qwen_proxy_raw.py --port 8080
QWEN_JWT="eyJ..." python3 -m qwen_proxy.raw --port 8080   # packaged equivalent

# After `pip install -e .`, the console script also runs SERVER mode (not raw):
qwen-proxy --port 8080

# Tests — unittest only (pytest is NOT installed). Run from the project root.
python3 -m unittest discover -s tests                       # full suite (~68 tests)
python3 -m unittest tests.test_toolcall                     # one module
python3 -m unittest tests.test_toolcall.ToolCallParserTests.test_parses_dsml_tool_call  # one test

# Smoke-check a running proxy
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
```

There is no linter/formatter configured and no pytest config. Tests are plain `unittest.TestCase`, import the package directly (`from qwen_proxy.toolcall import ...`), and need no live JWT — they exercise parsing/prompt logic offline.

The **JWT** (`QWEN_JWT` env or `--jwt`) is a browser bearer token: log into `chat.qwen.ai`, open DevTools → Network, send a message, copy `Authorization: Bearer <token>` from a `/api/v2/...` request. Expires ~30 days.

## How imports resolve (important)

The package source lives in `src/qwen_proxy/`, but tests and the root entrypoints `import qwen_proxy` **without** installing it. This works because the root `qwen_proxy.py` is a shim: it inserts `src/` on `sys.path` and sets `__path__ = ["src/qwen_proxy"]`, so `import qwen_proxy` and `from qwen_proxy.server import ...` both resolve into the package even though `qwen_proxy.py` is a single file. Always run commands from the project root so this shim is found first.

## Architecture

The whole proxy is one request-translation pipeline. Read `server.py` first — it owns routing, prompt rendering, the Qwen transport, and all four response handlers.

| File | Role |
|------|------|
| `src/qwen_proxy/server.py` | **Core.** `ProxyHandler` (stdlib `http.server`), routing, model resolution, `flatten_messages`, Qwen `create_chat`/payload/SSE transport, and the OpenAI + Anthropic × stream + non-stream handlers. |
| `src/qwen_proxy/raw.py` | `RawProxyHandler(ProxyHandler)` — overrides the handlers to **buffer the full upstream stream**, auto-retry on empty responses, and enable tool recovery. Monkey-patches `server.select_upstream_model` / `TOOL_RECOVERY` at import. |
| `src/qwen_proxy/toolcall.py` | Non-streaming tool-call parsing: DSML/XML (primary) → legacy `<\|tool_call\|>` → JSON (fallbacks), schema-aware arg normalization/validation, and context-based recovery (`infer_tool_calls_from_context`). |
| `src/qwen_proxy/toolstream.py` | `ToolStreamSieve` — buffers possibly-tool-call text during streaming so raw markup never leaks to the client as visible text; emits parsed `tool_calls` once a complete block is seen. |
| `src/qwen_proxy/anthropic.py` | Pure functions mapping Anthropic Messages requests → the internal OpenAI shape and OpenAI responses → Anthropic message/SSE blocks. No network. |
| `src/qwen_proxy/logging_utils.py` | `log_event(type, **fields)` — single-line JSON diagnostics to **stderr** (grep/jq-friendly). |
| `qwen_proxy.py`, `qwen_proxy_raw.py` | Root entrypoint shims (see import note). `qwen_proxy.py` → server mode; `qwen_proxy_raw.py` → raw mode. |

### The central design: tool calling is prompt-emulated

Qwen web has **no native function calling**. The proxy makes it work by prompt engineering + parsing:

1. `flatten_messages()` collapses the OpenAI/Anthropic `messages` array into a single text prompt with `[System] / [User] / [Assistant] / [Tool Result] / [System Reminder]` sections. When `tools` are present it prepends `TOOL_CALL_INSTRUCTION` (asking the model to emit a `<tool_calls><invoke name=…><parameter…>` XML/DSML block) plus a rendered tool list, and appends a recency-biased reminder.
2. The model's reply is scanned for that markup. `toolcall.parse_tool_calls()` (non-stream) or `ToolStreamSieve` (stream) converts it back into protocol-native `tool_calls` (OpenAI) or `tool_use` blocks (Anthropic).
3. **The proxy never executes tools.** It only translates markup into tool-call responses; the client/agent runner executes and sends `role:"tool"` results back, which `flatten_messages` re-renders as `[Tool Result]` context. This is a hard security boundary (see `ROADMAP.md`).

### Model resolution & thinking modes

A client model id like `qwen3.6-plus-thinking` is parsed by `resolve_model()` into `(qwen_upstream_id, thinking_mode)`:
- Suffix `-thinking` = always reason, `-fast` = never reason, no suffix = `auto`. The mode maps to Qwen's `feature_config` in `build_qwen_payload()`.
- The base name is mapped through `MODEL_ALIASES` + `BASE_MODELS` to the real upstream id (e.g. `qwen3.6-max` → `qwen3.6-max-preview`).
- `MODELS_LIST` (served at `/v1/models`) is the cross-product of `MODEL_LIST_BASE_IDS` × `{"", "-thinking", "-fast"}`.

`select_upstream_model(client_model, has_tools)` decides what actually hits Qwen: non-tool requests use the client's model; **tool requests are forced to the stable `TOOL_MODEL` (`qwen3.6-max-preview`)** because it parses tool markup reliably. Note this means a tool request loses the client's thinking suffix. The HTTP response still echoes the client's requested model name.

### server mode vs raw mode

Both now use stable-tool-fallback model selection. The difference is the streaming/recovery behavior:
- **server** (`server.py`, default): true passthrough streaming via the sieve; `RETRY_ON_FAIL` and `TOOL_RECOVERY` **off** by default. The proxy translates model-emitted markup and does not invent tool calls.
- **raw** (`raw.py`): collects the entire upstream stream into a buffer first, **retries up to `QWEN_RAW_MAX_RETRIES` times when Qwen returns an empty stream** (a known `qwen3.7-max` quirk), and enables `TOOL_RECOVERY`. Higher latency, more resilient.

### Qwen SSE handling

Upstream is always streamed (`stream:true`, `incremental_output:true`). Each `data:` event carries `choices[0].delta` with a `phase`: `thinking_summary` events are dropped (or accumulated as `thinking` in raw mode), `answer`/empty-phase events are the user-visible content. `usage` carries token counts. Every request creates a **new chat** (`create_chat`) — the proxy is stateless unless `QWEN_SESSION_TTL > 0` enables per-model session reuse.

## Configuration (environment variables)

| Var | Default | Effect |
|-----|---------|--------|
| `QWEN_JWT` | — | **Required** bearer token (or `--jwt`). |
| `QWEN_DEFAULT_MODEL` | `qwen3.6-plus` | Model when the request omits one. |
| `QWEN_TOOL_MODEL` / `QWEN_DEFAULT_TOOL_MODEL` | `qwen3.6-max-preview` | Stable upstream model for tool requests. Set to `none`/`request`/`client` to use the client's model instead. |
| `QWEN_RAW_TOOL_MODEL` | `qwen3.6-max-preview` | Raw-mode equivalent; `none` = true raw (client model for tools, accepting `Tool X does not exist` risk). |
| `QWEN_RAW_MAX_RETRIES` | `2` | Raw-mode empty-response retries. |
| `QWEN_RETRY_ON_FAIL` | `0` | server mode: retry once with a simplified prompt after a tool-parse miss. |
| `QWEN_TOOL_RECOVERY` | `0` | server mode: allow `infer_tool_calls_from_context` to synthesize a tool call when the model emits prose instead of markup. |
| `QWEN_SESSION_TTL` | `0` | Seconds to reuse a Qwen `chat_id` per model. `0` = stateless (new chat per request). |
| `QWEN_MAX_PROMPT_CHARS` | `0` | Intentionally disabled — see Gotchas. |

## Gotchas / invariants (don't "fix" these without reason)

- **Prompt truncation is deliberately off.** `_truncate_conversation` is a no-op; `QWEN_MAX_PROMPT_CHARS` only logs. Claude Code owns context compaction, and dropping messages here breaks tool-result continuity.
- **Recovery/retry are deliberately off in server mode.** The proxy's job is to translate model-emitted markup, not hallucinate tool calls from surrounding text.
- **Pentest-specific heuristics are hardcoded.** `_infer_skill_name` only recognizes `pentest-assistant-reasoning`, and recovery has special cases for `WebSearch`/`Skill`/browser-navigation tools. These are tuned for the specific agent runner this proxy fronts, not general logic.
- **All diagnostics go to stderr as JSON.** When debugging tool parsing, grep stderr for events like `tool_parse_ok`, `sieve_drop`, `tool_recovery`, `tool_miss`, `raw_retry` (see `logging_utils.log_event` docstring for the full list).
- **`raw.py` duplicates `qwen_proxy_raw.py`** (packaged vs root-shim copies). Keep handler changes in sync, or refactor the root shim to delegate.
- Root-level `test_*.py`, `qwen_debug.py`, `qwen_research.py`, and `scripts/claude_code_stress.py` are **manual/integration scripts** that hit a live proxy or the real Qwen API — they are not part of the `unittest` suite under `tests/`.
