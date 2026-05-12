import json
import unittest

from qwen_proxy import anthropic


class AnthropicAdapterTests(unittest.TestCase):
    def test_converts_anthropic_tools_to_openai_tools(self):
        req = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "open youtube"}],
            "tools": [{
                "name": "browser_open",
                "description": "Open URL",
                "input_schema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                },
            }],
        }

        out = anthropic.to_openai_request(req, "qwen3.6-plus")

        self.assertEqual(out["messages"][0], {"role": "user", "content": "open youtube"})
        self.assertEqual(out["tools"][0]["function"]["name"], "browser_open")
        self.assertEqual(out["tools"][0]["function"]["parameters"]["properties"]["url"]["type"], "string")

    def test_converts_tool_use_and_tool_result_loop(self):
        req = {
            "model": "claude-sonnet-4-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "browser_open",
                        "input": {"url": "https://youtube.com"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "opened",
                    }],
                },
            ],
        }

        out = anthropic.to_openai_request(req, "qwen3.6-plus")

        self.assertEqual(out["messages"][0]["role"], "assistant")
        self.assertEqual(out["messages"][0]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(out["messages"][1], {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": "opened",
        })

    def test_converts_openai_tool_call_to_anthropic_tool_use(self):
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "browser_open",
                            "arguments": '{"url":"https://youtube.com"}',
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }

        out = anthropic.from_openai_response(resp, "claude-sonnet-4-5")

        self.assertEqual(out["stop_reason"], "tool_use")
        self.assertEqual(out["content"][0]["type"], "tool_use")
        self.assertEqual(out["content"][0]["name"], "browser_open")
        self.assertEqual(out["content"][0]["input"], {"url": "https://youtube.com"})

    def test_stream_tool_use_block_shape(self):
        raw = anthropic.stream_tool_use_block(0, {
            "id": "call_1",
            "function": {"name": "run_shell_command", "arguments": '{"command":"date"}'},
        }).decode()

        self.assertIn("event: content_block_start", raw)
        self.assertIn('"type": "tool_use"', raw)
        self.assertIn('"name": "run_shell_command"', raw)
        self.assertIn('"command": "date"', raw)


if __name__ == "__main__":
    unittest.main()
