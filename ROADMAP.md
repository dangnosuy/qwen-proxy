# Qwen Proxy Refactor Roadmap

## Goal

Turn the current Qwen website reverse proxy into a standalone, OpenAI-compatible project that handles normal chat, streaming, tool calls, and tool-result continuation predictably.

The main target clients are agent runners such as Claude Code, OpenClaw, Codex-like CLIs, LiteLLM, aider/Continue-style integrations, and any client that expects OpenAI/Claude tool-call semantics. The proxy should never execute local tools itself; it should translate Qwen website output into protocol-correct tool-call responses so the agent runner can execute tools and send tool results back.

## What to borrow from ds2api

`ds2api` is useful because it treats website-output tool calling as a protocol translation problem instead of only a prompt trick.

- Separate layers: upstream website client, OpenAI HTTP surface, prompt rendering, SSE parsing, assistant-turn finalization, and tool-call parsing.
- Prefer a dedicated tool-call markup format. `ds2api` prompts the upstream model to emit DSML/XML blocks such as `<|DSML|tool_calls>` and converts them into OpenAI `tool_calls`.
- Use a streaming sieve. During stream mode, it buffers possible tool markup until it can decide whether the content is a real tool call or normal text, preventing raw tool syntax from leaking to the client.
- Normalize parsed arguments against the declared tool schema. If a tool field is declared as `string`, structured or numeric model output is coerced to a JSON string where needed.
- Treat final assistant output as a semantic object: text, thinking, tool calls, usage, finish reason, and validation errors are computed in one place for both stream and non-stream.

## Target Architecture

```text
qwen_proxy/
  pyproject.toml
  qwen_proxy.py                 # compatibility wrapper
  src/qwen_proxy/
    server.py                   # temporary legacy server entrypoint
    config.py                   # model/auth/runtime settings
    openai_types.py             # request/response helpers
    qwen_client.py              # Qwen chat create/completion transport
    prompt.py                   # message flattening and tool prompt rendering
    toolcall.py                 # DSML/XML + JSON fallback parser
    toolstream.py               # stream sieve for tool-call anti-leak
    assistant_turn.py           # finalization shared by stream/non-stream
    app.py                      # HTTP routes
  tests/
    test_toolcall.py
    test_toolstream.py
    test_prompt.py
    test_openai_shapes.py
```

## Implementation Plan

1. Done: Project split. The old `qwen_proxy.py` command still works, while the real server now lives under `src/qwen_proxy`.
2. In progress: Baseline tests. Parser tests exist and do not need a live Qwen JWT; response-shape tests still need to be added.
3. In progress: Tool-call protocol. The server now prompts for DSML/XML as the primary format, while keeping JSON `tool_calls` as fallback.
4. In progress: Parser hardening. The parser supports DSML/XML, canonical `<tool_calls>`, JSON fallback, and basic schema-aware argument normalization.
5. Done: Streaming sieve. Stream normal text immediately, buffer possible tool-call markup, emit OpenAI `delta.tool_calls` only when a full call is parsed.
6. In progress: Assistant-turn finalizer. Prompt history now renders previous `assistant.tool_calls` and `tool` results as structured context; full finalizer module still needs to centralize text/thinking/tool/usage/finish_reason.
7. Pending: Stateful continuation. Optionally reuse Qwen `chat_id` and `parent_id` for tool-result turns instead of creating a new chat for every request.
8. Pending: Operational polish. Add config file/env handling, better errors, health metadata, trace logging, and raw upstream capture for debugging.

## Review Checklist

- OpenAI compatibility: `/v1/models`, `/v1/chat/completions`, stream chunks, `finish_reason`, `usage`, CORS, error shape.
- Claude compatibility: `/v1/messages` and `/anthropic/v1/messages` adapter for Claude Code-style clients that do not speak OpenAI chat completions directly.
- Tool calling: `tools`, `tool_choice`, parallel calls, zero-arg calls, malformed arguments, legacy `<|tool_call|>` blocks, multi-turn tool result messages.
- Qwen-specific behavior: thinking phases, empty output, auth expiry, rate-limit shape, upstream SSE quirks, `parent_id` continuation.
- Client compatibility: OpenAI Python SDK, LiteLLM, Continue/aider-like clients, OpenCode/Qwen Code style tool schemas.
- Security boundary: proxy must only return tool-call requests; actual tool execution remains the client/agent framework's responsibility.

## Agent Loop Contract

For OpenAI-compatible clients, the intended loop is:

1. Client sends `messages` plus `tools` to `/v1/chat/completions`.
2. Qwen emits DSML/XML, legacy `<|tool_call|>`, or JSON `tool_calls` text.
3. Proxy converts that text into protocol-native `message.tool_calls` or streaming `delta.tool_calls`.
4. Client executes the tool locally.
5. Client sends the next request with the previous assistant `tool_calls` message and one or more `role: "tool"` result messages.
6. Proxy renders that history back to Qwen as structured context so the model can continue or call the next tool.
