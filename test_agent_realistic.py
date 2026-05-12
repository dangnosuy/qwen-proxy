#!/usr/bin/env python3
"""
Simulate realistic agent framework requests — the kind OpenCode/Qwen Code CLI sends.
Big system prompt, many tools, complex user requests.
"""

import json
import sys
import urllib.request

PROXY = "http://localhost:9090/v1/chat/completions"
MODEL = "qwen3.6-plus"

AGENT_SYSTEM_PROMPT = """You are an AI coding assistant. You help users with software engineering tasks.
You have access to tools to interact with the local filesystem and execute commands.
Use tools when needed to accomplish the user's request.
Always use the appropriate tool — never pretend to execute commands or read files yourself.
When you need to see file contents, use the ReadFile tool.
When you need to run a command, use the run_shell_command tool.
When you need to list directory contents, use the list_directory tool.
When you need to write files, use the write_file tool.
When you need to search for files, use the glob tool.
Be concise and helpful."""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ReadFile",
            "description": "Read the contents of a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The absolute path to the file to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Execute a shell command and return stdout and stderr",
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
            "name": "list_directory",
            "description": "List files and directories in the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The directory path to list"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if it doesn't exist",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to write to"},
                    "content": {"type": "string", "description": "The content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Search for files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The glob pattern to match (e.g. **/*.py)"},
                    "path": {"type": "string", "description": "The base directory to search from"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]


def call(messages, stream=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": AGENT_TOOLS,
        "stream": stream,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(PROXY, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if stream:
            full = ""
            tool_calls = []
            finish = None
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                evt = json.loads(line[6:])
                delta = evt["choices"][0].get("delta", {})
                finish = evt["choices"][0].get("finish_reason") or finish
                if "content" in delta and delta["content"]:
                    full += delta["content"]
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        tool_calls.append(tc)
            return {"content": full, "tool_calls": tool_calls, "finish_reason": finish}
        else:
            result = json.loads(resp.read())
            msg = result["choices"][0]["message"]
            return {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", []),
                "finish_reason": result["choices"][0]["finish_reason"],
            }


def show_result(name, result):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"  finish_reason: {result['finish_reason']}")
    if result["tool_calls"]:
        for tc in result["tool_calls"]:
            fn = tc.get("function", {})
            print(f"  TOOL CALL: {fn.get('name', '?')}({fn.get('arguments', '{}')[:100]})")
        print(f"  -> PASS")
    elif result["content"]:
        preview = result["content"][:200].replace("\n", "\\n")
        print(f"  CONTENT: {preview}")
        if "does not exist" in result["content"].lower():
            print(f"  -> FAIL: model thinks tool doesn't exist")
        else:
            print(f"  -> TEXT (may or may not be correct)")
    else:
        print(f"  -> EMPTY response")


msgs = lambda user_content: [
    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    {"role": "user", "content": user_content},
]

# ============================================================
# TEST 1: "list files in current directory" (non-stream)
# ============================================================
r = call(msgs("ở trong thư mục /tmp đang có gì?"))
show_result("list_directory /tmp (non-stream)", r)

# ============================================================
# TEST 2: "read a file" (non-stream)
# ============================================================
r = call(msgs("đọc file /etc/hostname cho tôi"))
show_result("ReadFile /etc/hostname (non-stream)", r)

# ============================================================
# TEST 3: "write a file" (non-stream)
# ============================================================
r = call(msgs("viết nội dung 'hello world' vào file /tmp/test_output.txt"))
show_result("write_file (non-stream)", r)

# ============================================================
# TEST 4: "search for python files" (non-stream)
# ============================================================
r = call(msgs("tìm tất cả file python trong /home/dangnosuy/kaggle"))
show_result("glob *.py (non-stream)", r)

# ============================================================
# TEST 5: "run a command" (non-stream)
# ============================================================
r = call(msgs("chạy lệnh 'uname -a' cho tôi"))
show_result("run_shell_command uname (non-stream)", r)

# ============================================================
# TEST 6: Same but STREAMING
# ============================================================
r = call(msgs("đọc file /etc/os-release"), stream=True)
show_result("ReadFile /etc/os-release (STREAM)", r)

r = call(msgs("liệt kê các file trong /tmp"), stream=True)
show_result("list_directory /tmp (STREAM)", r)

# ============================================================
# TEST 7: Complex multi-step (web search + write)
# ============================================================
r = call(msgs("hãy search cho tôi trang web vnexpress.vn và tìm kiếm bài viết mới nhất, sau đó viết báo cáo vào /tmp/report.md"))
show_result("web_search + write_file (complex)", r)

# ============================================================
# TEST 8: Multi-turn conversation with tool results
# ============================================================
print(f"\n{'='*60}")
print(f"TEST: Multi-turn: list dir -> get result -> read file")
print(f"{'='*60}")

# Turn 1: ask to explore
r1 = call([
    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    {"role": "user", "content": "trong thư mục /home/dangnosuy/kaggle có những file gì?"},
])
print(f"  Turn 1 finish: {r1['finish_reason']}")
if r1["tool_calls"]:
    tc1 = r1["tool_calls"][0]
    fn1 = tc1.get("function", {})
    print(f"  Turn 1 TOOL: {fn1.get('name')}({fn1.get('arguments', '')[:80]})")

    # Turn 2: provide tool result, ask follow-up
    r2 = call([
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": "trong thư mục /home/dangnosuy/kaggle có những file gì?"},
        {"role": "assistant", "content": None, "tool_calls": [tc1]},
        {"role": "tool", "tool_call_id": tc1.get("id", "call_1"), "content": "qwen_proxy/\nqwen_proxy/qwen_proxy.py\nqwen_proxy/qwen_debug.py\nqwen_proxy/qwen_research.py\nqwen_proxy/QWEN_API_RESEARCH.md\nqwen_proxy/QWEN_PROXY_GUIDE.md\nqwen_proxy/test_tool_calling.py\nqwen_proxy/test_agent_realistic.py"},
        {"role": "user", "content": "đọc file qwen_proxy/qwen_proxy.py cho tôi"},
    ])
    print(f"  Turn 2 finish: {r2['finish_reason']}")
    if r2["tool_calls"]:
        fn2 = r2["tool_calls"][0].get("function", {})
        print(f"  Turn 2 TOOL: {fn2.get('name')}({fn2.get('arguments', '')[:80]})")
        print(f"  -> PASS: multi-turn tool calling works")
    elif r2["content"]:
        print(f"  Turn 2 CONTENT: {r2['content'][:150]}")
    else:
        print(f"  Turn 2 EMPTY")
else:
    print(f"  Turn 1 CONTENT: {r1['content'][:150]}")
    print(f"  -> FAIL: no tool call in turn 1")

# ============================================================
# TEST 9: No tool needed — just greeting
# ============================================================
r = call(msgs("xin chào, bạn khỏe không?"))
show_result("Greeting — no tool needed", r)

print(f"\n{'='*60}")
print("ALL TESTS DONE")
print(f"{'='*60}")
