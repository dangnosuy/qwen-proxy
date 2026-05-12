#!/usr/bin/env python3
"""Test tool calling qua qwen_proxy. Chạy khi proxy đang listen trên port 9090."""

import json
import sys
import urllib.request

PROXY = "http://localhost:9090/v1/chat/completions"
MODEL = "qwen3.6-plus"


def call(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(PROXY, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def test(name: str, payload: dict):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        result = call(payload)
        msg = result["choices"][0]["message"]
        finish = result["choices"][0]["finish_reason"]
        print(f"  finish_reason: {finish}")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                print(f"  TOOL CALL: {fn['name']}({fn['arguments']})")
            print(f"  -> OK: model returned tool_calls")
        else:
            content = msg.get("content", "")
            preview = content[:200].replace("\n", "\\n")
            print(f"  CONTENT: {preview}")
            if "does not exist" in content.lower():
                print(f"  -> FAIL: model said tool doesn't exist")
            else:
                print(f"  -> OK: text response (no tool call)")
    except Exception as e:
        print(f"  ERROR: {e}")


# ============================================================
# TEST 1: Single tool — get_current_time
# ============================================================
test("Single tool: get_current_time", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "What time is it right now?"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
})

# ============================================================
# TEST 2: Single tool — read_file
# ============================================================
test("Single tool: read_file", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Read the file /etc/hostname"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read"}
                },
                "required": ["path"]
            }
        }
    }]
})

# ============================================================
# TEST 3: Single tool — run_shell_command
# ============================================================
test("Single tool: run_shell_command", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "List all files in the current directory"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Execute a shell command and return its output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    }]
})

# ============================================================
# TEST 4: Single tool — write_file
# ============================================================
test("Single tool: write_file", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Write 'hello world' to /tmp/test.txt"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to write"},
                    "content": {"type": "string", "description": "The content to write"}
                },
                "required": ["path", "content"]
            }
        }
    }]
})

# ============================================================
# TEST 5: Multiple tools available
# ============================================================
test("Multiple tools: user asks to list files", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "List all Python files in /home"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Execute a shell command and return its output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to execute"}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file at the given path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The file path to read"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file at the given path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The file path to write"},
                        "content": {"type": "string", "description": "The content to write"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ]
})

# ============================================================
# TEST 6: Multi-turn with tool result
# ============================================================
print(f"\n{'='*60}")
print(f"TEST: Multi-turn: tool call -> tool result -> followup")
print(f"{'='*60}")

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "parameters": {"type": "object", "properties": {}}
    }
}]

# Turn 1: expect tool call
r1 = call({
    "model": MODEL,
    "messages": [{"role": "user", "content": "What time is it?"}],
    "tools": tools,
})
msg1 = r1["choices"][0]["message"]
print(f"  Turn 1 finish_reason: {r1['choices'][0]['finish_reason']}")

if msg1.get("tool_calls"):
    tc = msg1["tool_calls"][0]
    print(f"  Turn 1 TOOL CALL: {tc['function']['name']}")

    # Turn 2: send tool result back
    r2 = call({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": tc["id"], "content": "2026-04-30 20:45:00 UTC"},
        ],
        "tools": tools,
    })
    msg2 = r2["choices"][0]["message"]
    print(f"  Turn 2 finish_reason: {r2['choices'][0]['finish_reason']}")
    content2 = msg2.get("content", "")
    print(f"  Turn 2 CONTENT: {content2[:200]}")
    if msg2.get("tool_calls"):
        print(f"  Turn 2 called tools again (unexpected)")
    else:
        print(f"  -> OK: model used tool result to answer")
else:
    content1 = msg1.get("content", "")[:200]
    print(f"  Turn 1 CONTENT: {content1}")
    print(f"  -> FAIL: model didn't call tool")

# ============================================================
# TEST 7: No tool needed — should respond with text
# ============================================================
test("No tool needed: greeting", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Execute a shell command and return its output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    }]
})

print(f"\n{'='*60}")
print("ALL TESTS DONE")
print(f"{'='*60}")
