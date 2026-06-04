import unittest

from qwen_proxy.server import _sanitize_assistant_text, _truncate_conversation, flatten_messages


class PromptRenderingTests(unittest.TestCase):
    def test_renders_tool_loop_history_as_structured_context(self):
        prompt = flatten_messages([
            {"role": "user", "content": "open youtube"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "browser_open",
                        "arguments": '{"url":"https://www.youtube.com","newTab":true}',
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "opened"},
            {"role": "user", "content": "play a song"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "browser_open",
                "description": "Open a browser URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                },
            },
        }])

        self.assertIn("<tool_calls>", prompt)
        self.assertIn('<invoke name="browser_open">', prompt)
        self.assertIn("Tool name: browser_open", prompt)
        self.assertIn("Result:\nopened", prompt)
        self.assertIn("request exactly one next client capability", prompt)
        self.assertNotIn("call the tool NOW", prompt)

    def test_tool_result_tail_reminder_prefers_answering(self):
        prompt = flatten_messages([
            {"role": "user", "content": "what time is it?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "10:30"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get time",
                "parameters": {"type": "object", "properties": {}},
            },
        }])

        self.assertIn("You just received a tool result", prompt)
        self.assertIn("Use it to answer normally", prompt)
        self.assertNotIn("call the tool NOW", prompt)

    def test_renders_tool_choice_required(self):
        prompt = flatten_messages([
            {"role": "user", "content": "what time is it?"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get time",
                "parameters": {"type": "object", "properties": {}},
            },
        }], tool_choice="required")

        self.assertIn("Tool choice: you MUST serialize one available client capability request.", prompt)

    def test_renders_forced_tool_choice(self):
        prompt = flatten_messages([
            {"role": "user", "content": "open youtube"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "browser_open",
                "description": "Open URL",
                "parameters": {"type": "object", "properties": {}},
            },
        }], tool_choice={"type": "function", "function": {"name": "browser_open"}})

        self.assertIn("Tool choice: you MUST serialize a request for the client capability named browser_open.", prompt)

    def test_auto_tool_reminder_does_not_force_tool_call_now(self):
        prompt = flatten_messages([
            {"role": "user", "content": "hello"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Run command",
                "parameters": {"type": "object", "properties": {}},
            },
        }])

        self.assertIn("If the next step requires an action", prompt)
        self.assertIn("If no action is needed, answer normally", prompt)
        self.assertNotIn("call the tool NOW", prompt)

    def test_long_agent_system_prompt_is_not_compacted(self):
        system = (
            "You are Claude Code, Anthropic's official CLI for Claude.\n"
            "Current working directory: /tmp/qwen_proxy_claude_stress_fixture\n"
            "KEEP_THIS_TAIL_MARKER\n"
            + ("tool_use function calling coding assistant " * 120)
            + "\nEND_OF_LONG_SYSTEM_PROMPT"
        )
        prompt = flatten_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": "read notes/ops.md"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        }])

        self.assertIn("Current working directory: /tmp/qwen_proxy_claude_stress_fixture", prompt)
        self.assertIn("KEEP_THIS_TAIL_MARKER", prompt)
        self.assertIn("END_OF_LONG_SYSTEM_PROMPT", prompt)
        self.assertNotIn("System context truncated", prompt)
        self.assertNotIn("CLI coding assistant. Follow the user's request", prompt)

    def test_qwen_code_context_is_not_compacted(self):
        context = (
            "This is the Qwen Code. We are setting up the context for our chat.\n"
            "Today's date is 2026-05-28.\n"
            "My operating system is: Linux.\n"
            "I'm currently working in the directory: /tmp/project.\n"
            "KEEP_FULL_CONTEXT"
        )
        prompt = flatten_messages([
            {"role": "user", "content": context},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        }])

        self.assertIn("This is the Qwen Code", prompt)
        self.assertIn("KEEP_FULL_CONTEXT", prompt)
        self.assertNotIn("Qwen Code session context initialized", prompt)

    def test_web_search_request_gets_explicit_tool_hint(self):
        prompt = flatten_messages([
            {"role": "user", "content": "lên google tìm vnexpress bài mới nhất"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }])

        self.assertIn("The latest user request is a web search", prompt)
        self.assertIn("Use WebSearch first", prompt)
        self.assertIn("Do not answer from memory", prompt)

    def test_local_search_request_does_not_get_web_search_hint(self):
        prompt = flatten_messages([
            {
                "role": "user",
                "content": (
                    "dùng phantom_rce.py đọc cấu trúc thư mục và bắt đầu tìm kiếm "
                    "phiên bản production trong codebase"
                ),
            },
        ], tools=[{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }, {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a local file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        }])

        self.assertNotIn("The latest user request is a web search", prompt)
        self.assertNotIn("Use WebSearch first", prompt)

    def test_current_info_words_without_explicit_web_do_not_force_web_search(self):
        prompt = flatten_messages([
            {"role": "user", "content": "tìm phiên bản production mới nhất đang chạy"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }])

        self.assertNotIn("The latest user request is a web search", prompt)
        self.assertNotIn("Use WebSearch first", prompt)

    def test_proxy_does_not_truncate_claude_code_history(self):
        messages = [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "first user"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path":"/tmp/a"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
            {"role": "user", "content": "continue"},
        ]

        self.assertIs(_truncate_conversation(messages, tools=None, max_chars=10), messages)

    def test_empty_web_search_result_gets_one_retry_hint(self):
        prompt = flatten_messages([
            {"role": "user", "content": "google tìm vnexpress bài mới nhất"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "WebSearch", "arguments": '{"query":"vnexpress bài mới nhất"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Did 0 searches in 4s"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }])

        self.assertIn("Retry WebSearch once", prompt)
        self.assertIn("broader query", prompt)
        self.assertIn("Do not use other capability names", prompt)

    def test_orphan_empty_web_search_result_is_recognized(self):
        prompt = flatten_messages([
            {"role": "user", "content": "kiểm tra chính sách macro hiện nay"},
            {
                "role": "tool",
                "tool_call_id": "call_missing",
                "content": (
                    'Web search results for query: "Microsoft Office macro policy 2025"\n\n\n'
                    "REMINDER: You MUST include the sources above in your response."
                ),
            },
        ], tools=[{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }])

        self.assertIn("Retry WebSearch once", prompt)
        self.assertNotIn("The latest user request is a web search", prompt)

    def test_repeated_orphan_empty_web_search_results_stop_retrying(self):
        prompt = flatten_messages([
            {"role": "user", "content": "kiểm tra chính sách macro hiện nay"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": 'Web search results for query: "q1"\n\n\nREMINDER: cite sources.',
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": 'Web search results for query: "q2"\n\n\nREMINDER: cite sources.',
            },
        ], tools=[{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }])

        self.assertIn("already returned no results", prompt)
        self.assertIn("Answer honestly", prompt)
        self.assertNotIn("Retry WebSearch once", prompt)

    def test_sanitizes_tool_does_not_exist_artifact_from_text(self):
        cleaned = _sanitize_assistant_text(
            "Tool mcp__fetch__fetch does not exists.Dưới đây là kết quả.",
            tools=[{
                "type": "function",
                "function": {"name": "mcp__fetch__fetch", "parameters": {"type": "object"}},
            }],
        )

        self.assertEqual(cleaned, "Dưới đây là kết quả.")


if __name__ == "__main__":
    unittest.main()
