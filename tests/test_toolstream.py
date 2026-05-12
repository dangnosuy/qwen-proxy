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

    def test_drops_malformed_tool_markup_on_flush(self):
        sieve = ToolStreamSieve()
        sieve.feed('<|DSML|tool_calls><|DSML|invoke name="Bash">')
        events = sieve.flush()

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
