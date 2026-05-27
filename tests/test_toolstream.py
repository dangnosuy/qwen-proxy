import json
import unittest

from qwen_proxy.toolstream import ToolStreamSieve


class ToolStreamSieveTests(unittest.TestCase):
    def test_buffers_legacy_tool_call_until_complete(self):
        sieve = ToolStreamSieve()

        self.assertEqual(sieve.feed("<|tool_"), [])
        self.assertEqual(sieve.feed('call|>{"id":"browser_open","arguments":{"url":"https://x.test"}}'), [])
        events = sieve.feed("</|tool_call|>")

        self.assertEqual(len(events), 1)
        call = events[0].tool_calls[0]
        self.assertEqual(call["function"]["name"], "browser_open")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"url": "https://x.test"})

    def test_streams_normal_text_without_tool_marker(self):
        sieve = ToolStreamSieve()
        events = sieve.feed("hello " * 20)
        events += sieve.flush()

        self.assertEqual("".join(event.content for event in events), "hello " * 20)

    def test_emits_prefix_then_tool_call(self):
        sieve = ToolStreamSieve()
        events = sieve.feed('I need a tool.\n<tool_call>{"name":"run_shell_command","arguments":{"command":"date"}}</tool_call>')

        self.assertEqual(events[0].content, "I need a tool.\n")
        self.assertEqual(events[1].tool_calls[0]["function"]["name"], "run_shell_command")

    def test_buffers_json_tool_call_until_complete(self):
        sieve = ToolStreamSieve()

        self.assertEqual(sieve.feed('{"tool_calls":['), [])
        events = sieve.feed('{"name":"run_shell_command","arguments":{"command":"date"}}]}')

        call = events[0].tool_calls[0]
        self.assertEqual(call["function"]["name"], "run_shell_command")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "date"})

    def test_does_not_emit_repaired_dsml_before_stream_flush(self):
        sieve = ToolStreamSieve()
        chunk = """<|DSML|tool_calls>
  <|DSML|invoke name="run_shell_command">
    <|DSML|parameter name="command"><![CDATA[date]]></|DSML|parameter>
"""

        self.assertEqual(sieve.feed(chunk), [])

        events = sieve.flush()
        call = events[0].tool_calls[0]
        self.assertEqual(call["function"]["name"], "run_shell_command")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "date"})

    def test_flush_repairs_zero_argument_dsml_without_root_close(self):
        sieve = ToolStreamSieve()
        chunk = """<|DSML|tool_calls>
  <|DSML|invoke name="mcp__playwright__browser_take_screenshot">
  </|DSML|invoke>
"""

        self.assertEqual(sieve.feed(chunk), [])

        events = sieve.flush()
        call = events[0].tool_calls[0]
        self.assertEqual(call["function"]["name"], "mcp__playwright__browser_take_screenshot")
        self.assertEqual(json.loads(call["function"]["arguments"]), {})

    def test_drops_malformed_tool_markup_on_flush(self):
        sieve = ToolStreamSieve()
        sieve.feed('<|DSML|tool_calls><|DSML|invoke name="Bash">')
        events = sieve.flush()

        self.assertEqual(events, [])
        self.assertIn('name="Bash"', sieve.dropped_markup)

    def test_flush_repairs_qwen_parameter_equals_markup(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {"command": {"type": "string"}},
                },
            },
        }]
        sieve = ToolStreamSieve(tools)
        chunk = """<tool_calls>
  <invoke name="Bash">
<parameter=command>
rmpc status 2>&1
echo "---"
pgrep -a mpd
</parameter>
</invoke>
</tool_calls>"""

        self.assertEqual(sieve.feed(chunk), [])
        events = sieve.flush()
        call = events[0].tool_calls[0]
        args = json.loads(call["function"]["arguments"])

        self.assertEqual(call["function"]["name"], "Bash")
        self.assertIn("rmpc status", args["command"])


if __name__ == "__main__":
    unittest.main()
