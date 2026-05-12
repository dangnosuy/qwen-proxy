#!/usr/bin/env python3
"""Test complex/edge cases that previously failed."""

import json
import urllib.request

PROXY = "http://localhost:9090/v1/chat/completions"
MODEL = "qwen3.6-plus"

SYSTEM = """You are an AI coding assistant. You help users with software engineering tasks.
You have access to tools to interact with the local filesystem and execute commands.
Use tools when needed to accomplish the user's request.
Always use the appropriate tool — never pretend to execute commands or read files yourself.
When you need to see file contents, use the ReadFile tool.
When you need to run a command, use the run_shell_command tool.
When you need to list directory contents, use the list_directory tool.
When you need to write files, use the write_file tool.
When you need to search for files, use the glob tool.
Be concise and helpful."""

TOOLS = [
    {"type": "function", "function": {"name": "ReadFile", "description": "Read the contents of a file at the given path", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The absolute path to the file to read"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell_command", "description": "Execute a shell command and return stdout and stderr", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to execute"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "list_directory", "description": "List files and directories in the given path", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The directory path to list"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to a file, creating it if it doesn't exist", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The file path to write to"}, "content": {"type": "string", "description": "The content to write to the file"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Search for files matching a glob pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The glob pattern to match"}, "path": {"type": "string", "description": "The base directory to search from"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}},
]


def call(messages):
    data = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS}).encode()
    req = urllib.request.Request(PROXY, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    msg = result["choices"][0]["message"]
    return {
        "content": msg.get("content", ""),
        "tool_calls": msg.get("tool_calls", []),
        "finish_reason": result["choices"][0]["finish_reason"],
    }


def test(name, user_msg):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    r = call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ])
    print(f"  finish_reason: {r['finish_reason']}")
    if r["tool_calls"]:
        for tc in r["tool_calls"]:
            fn = tc["function"]
            print(f"  TOOL: {fn['name']}({fn['arguments'][:120]})")
        status = "PASS"
    else:
        preview = (r["content"] or "")[:200].replace("\n", "\\n")
        print(f"  TEXT: {preview}")
        if "does not exist" in (r["content"] or "").lower():
            status = "FAIL (hallucinated 'does not exist')"
        else:
            status = "TEXT (no tool call)"
    print(f"  -> {status}")
    return status


results = []

# Previously failing case
results.append(test(
    "web_search + write (complex multi-step)",
    "hãy search cho tôi trang web vnexpress.vn và viết báo cáo vào /tmp/report.md"
))

# More complex cases
results.append(test(
    "Explore directory then summarize",
    "xem trong /home/dangnosuy/kaggle có gì và giải thích cho tôi"
))

results.append(test(
    "Find and read specific file",
    "tìm file có tên qwen_proxy.py trong /home/dangnosuy/kaggle/qwen_proxy và đọc nó"
))

results.append(test(
    "Run command and write output",
    "chạy lệnh 'date' rồi ghi kết quả vào /tmp/date_output.txt"
))

results.append(test(
    "Vietnamese: liệt kê folder",
    "liệt kê các file và folder trong thư mục /home/dangnosuy/kaggle"
))

results.append(test(
    "English: what files are here",
    "what files are in /home/dangnosuy/kaggle?"
))

results.append(test(
    "Greeting (no tool needed)",
    "xin chào!"
))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for i, s in enumerate(results):
    print(f"  Test {i+1}: {s}")
pass_count = sum(1 for s in results if s == "PASS")
print(f"\n  {pass_count}/{len(results)} PASS")
