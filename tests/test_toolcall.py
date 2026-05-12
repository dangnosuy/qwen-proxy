import json
import unittest

from qwen_proxy.toolcall import parse_tool_calls


class ToolCallParserTests(unittest.TestCase):
    def test_parses_dsml_tool_call(self):
        text = """
<|DSML|tool_calls>
  <|DSML|invoke name="read_file">
    <|DSML|parameter name="path"><![CDATA[/tmp/a.txt]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>
"""
        calls = parse_tool_calls(text)

        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/tmp/a.txt"})

    def test_parses_canonical_xml_tool_call(self):
        text = '<tool_calls><invoke name="noop"></invoke></tool_calls>'
        calls = parse_tool_calls(text)

        self.assertEqual(calls[0]["function"]["name"], "noop")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {})

    def test_keeps_json_tool_call_fallback(self):
        text = '```json\n{"tool_calls":[{"function":{"name":"search","arguments":{"query":"qwen"}}}]}\n```'
        calls = parse_tool_calls(text)

        self.assertEqual(calls[0]["function"]["name"], "search")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"query": "qwen"})

    def test_normalizes_declared_string_schema(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
            },
        }]
        text = """
<tool_calls>
  <invoke name="write_file">
    <parameter name="content">{"a": 1}</parameter>
  </invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(args["content"], '{"a": 1}')

    def test_parses_legacy_single_tool_call_tag_from_web_output(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "browser_open",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "url": {"type": "string"},
                        "newTab": {"type": "boolean"},
                    },
                },
            },
        }]
        text = """
<|tool_call|>
{"id": "browser_open", "arguments": {"action": "open", "url": "https://www.youtube.com (https://www.youtube.com/)", "newTab": true}}
</|tool_call>
"""
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "browser_open")
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(args["url"], "https://www.youtube.com")

    def test_parses_legacy_single_tool_call_with_name_field(self):
        text = '<tool_call>{"name":"run_shell_command","arguments":{"command":"date"}}</tool_call>'
        calls = parse_tool_calls(text)

        self.assertEqual(calls[0]["function"]["name"], "run_shell_command")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "date"})

    def test_normalizes_fullwidth_dsml_punctuation(self):
        text = """
<｜DSML｜tool_calls>
  <｜DSML｜invoke name="run_shell_command">
    <｜DSML｜parameter name="command"><![CDATA[date]]></｜DSML｜parameter>
  </｜DSML｜invoke>
</｜DSML｜tool_calls>
"""
        calls = parse_tool_calls(text)

        self.assertEqual(calls[0]["function"]["name"], "run_shell_command")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "date"})

    def test_rejects_missing_required_arguments(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {"command": {"type": "string"}},
                },
            },
        }]
        text = """
<tool_calls>
  <invoke name="run_shell_command">
    <parameter name="command"></parameter>
  </invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools)
        self.assertIsNone(calls)


if __name__ == "__main__":
    unittest.main()
