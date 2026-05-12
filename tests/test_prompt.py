import unittest

from qwen_proxy.server import flatten_messages


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

        self.assertIn("<|DSML|tool_calls>", prompt)
        self.assertIn('<|DSML|invoke name="browser_open">', prompt)
        self.assertIn("Tool name: browser_open", prompt)
        self.assertIn("Result:\nopened", prompt)
        self.assertIn("call exactly one next tool", prompt)

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

        self.assertIn("Tool choice: you MUST call one of the available tools.", prompt)

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

        self.assertIn("Tool choice: you MUST call the tool named browser_open.", prompt)


if __name__ == "__main__":
    unittest.main()
