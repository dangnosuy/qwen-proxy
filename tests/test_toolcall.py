import json
import unittest

from qwen_proxy.toolcall import infer_tool_calls_from_context, parse_tool_calls


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

    def test_repairs_truncated_zero_argument_dsml_tool_call(self):
        text = """
<|DSML|tool_calls>
  <|DSML|invoke name="mcp__playwright__browser_take_screenshot">
  </|DSML|invoke>
"""
        calls = parse_tool_calls(text)

        self.assertEqual(calls[0]["function"]["name"], "mcp__playwright__browser_take_screenshot")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {})

    def test_incomplete_repair_can_be_disabled_for_active_streams(self):
        text = """
<|DSML|tool_calls>
  <|DSML|invoke name="run_shell_command">
    <|DSML|parameter name="command"><![CDATA[date]]></|DSML|parameter>
"""
        calls = parse_tool_calls(text, allow_incomplete=False)

        self.assertIsNone(calls)

    def test_keeps_json_tool_call_fallback(self):
        text = '```json\n{"tool_calls":[{"function":{"name":"search","arguments":{"query":"qwen"}}}]}\n```'
        calls = parse_tool_calls(text)

        self.assertEqual(calls[0]["function"]["name"], "search")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"query": "qwen"})

    def test_repairs_json_tool_call_missing_call_object_close(self):
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
        text = r'''{"tool_calls":[{"name":"Bash","arguments":{"command":"curl -sk -c /tmp/weather_cookies.txt -L \"https://weather.nldc.evn.vn/Secure/Login.aspx\" 2>/dev/null | grep -oP 'id=\"__VIEWSTATE\"[^>]*value=\"([^\"]*)\"' | head -1","id":"call_a1b2c3d4e5f6"}]}'''
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["id"], "call_a1b2c3d4e5f6")
        self.assertEqual(calls[0]["function"]["name"], "Bash")
        self.assertIn("weather.nldc.evn.vn", args["command"])
        self.assertNotIn("id", args)

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

    def test_repairs_qwen_parameter_equals_command_markup(self):
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
        text = """
<tool_calls>
  <invoke name="Bash">
<parameter=command>
rmpc status 2>&1
echo "---"
pgrep -a mpd
</parameter>
</invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "Bash")
        self.assertIn("rmpc status", args["command"])
        self.assertIn("pgrep -a mpd", args["command"])

    def test_maps_qwen_bash_code_parameter_to_required_command(self):
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
        text = """
<tool_calls>
  <invoke name="Bash">
<parameter=code><![CDATA[
cat > ~/.config/mpd/mpd.conf <<EOF
music_directory "$HOME/Music"
EOF
]]></parameter>
</invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "Bash")
        self.assertIn("mpd.conf", args["command"])
        self.assertNotIn("code", args)

    def test_parses_direct_argument_tags_inside_invoke(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "old_string", "new_string"],
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                },
            },
        }]
        text = """
Giờ cập nhật run_round:

<tool_calls>
  <invoke name="Edit">
    <parameter name="file_path"><![CDATA[/home/dangnosuy/Documents/LaiXe/moodle_quiz_auto.py]]></parameter>
    <old_string><![CDATA[answered = save_all_answers(fixed_answer=fixed_answer)]]></old_string>
    <new_string><![CDATA[answered = save_all_answers(fixed_answer=fixed_answer, total_slots=TOTAL_SLOTS)]]></new_string>
  </invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "Edit")
        self.assertEqual(args["file_path"], "/home/dangnosuy/Documents/LaiXe/moodle_quiz_auto.py")
        self.assertIn("total_slots=TOTAL_SLOTS", args["new_string"])
        self.assertIn("save_all_answers", args["old_string"])

    def test_repairs_truncated_parameter_equals_code_markup(self):
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
        text = '<tool_calls><invoke name="Bash"><parameter=code>echo truncated probe\n'
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(args["command"], "echo truncated probe")

    def test_maps_qwen_websearch_queries_parameter_to_query(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }]
        text = """
<tool_calls>
<invoke name="WebSearch">
<parameter=queries>
[
  "site:vnexpress.net tin mới nhất"
]
</parameter>
</invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "WebSearch")
        self.assertEqual(args["query"], "site:vnexpress.net tin mới nhất")
        self.assertNotIn("queries", args)

    def test_repairs_direct_query_tag_closed_as_parameter(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "WebSearch",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        }]
        text = """
<tool_calls>
  <invoke name="WebSearch">
    <query><![CDATA[Microsoft Office VBA macro security changes 2024 block internet macros MOTW]]></parameter>
  </invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "WebSearch")
        self.assertIn("Microsoft Office VBA", args["query"])

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

    def test_parses_qwen_malformed_dsml_closing_slash(self):
        text = """
<|DSML|tool_calls>
  <|DSML|invoke name="Skill">
    <|DSML|parameter name="skill"><![CDATA[pentest-assistant-reasoning]]></|DSML|parameter>
  </|/DSML|invoke>
</|DSML|tool_calls>
"""
        calls = parse_tool_calls(text)
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "Skill")
        self.assertEqual(args["skill"], "pentest-assistant-reasoning")

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

    def test_infers_pentest_reasoning_skill_from_context(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "Skill",
                "parameters": {
                    "type": "object",
                    "required": ["skill"],
                    "properties": {"skill": {"type": "string"}},
                },
            },
        }]
        text = """
<tool_calls>
  <invoke name="Skill"></invoke>
</tool_calls>
"""
        calls = parse_tool_calls(text, tools, context="load skill pentet assistant reassioning hiện tại")

        self.assertIsNone(calls)

    def test_can_repair_missing_skill_argument_when_explicitly_enabled(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "Skill",
                "parameters": {
                    "type": "object",
                    "required": ["skill"],
                    "properties": {"skill": {"type": "string"}},
                },
            },
        }]
        text = """
<tool_calls>
  <invoke name="Skill"></invoke>
</tool_calls>
"""
        calls = parse_tool_calls(
            text,
            tools,
            repair_missing=True,
            context="load skill pentet assistant reassioning hiện tại",
        )
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "Skill")
        self.assertEqual(args["skill"], "pentest-assistant-reasoning")

    def test_infers_skill_argument_from_schema_name_field(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "Skill",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}},
                },
            },
        }]
        calls = infer_tool_calls_from_context(
            tools,
            "load skill pentet assistant reassioning hiện tại",
            "Tool Skill does not exists.",
        )
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "Skill")
        self.assertEqual(args["name"], "pentest-assistant-reasoning")

    def test_fills_missing_browser_navigate_url_from_context(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "mcp__playwright__browser_navigate",
                "parameters": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {"url": {"type": "string"}},
                },
            },
        }]
        text = '<tool_calls><invoke name="mcp__playwright__browser_navigate"></invoke></tool_calls>'
        calls = parse_tool_calls(text, tools, context="Open https://pmis.evn.com.vn/qlkt/tmsLogin.jsf now")

        self.assertIsNone(calls)

    def test_can_repair_missing_browser_navigate_url_when_explicitly_enabled(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "mcp__playwright__browser_navigate",
                "parameters": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {"url": {"type": "string"}},
                },
            },
        }]
        text = '<tool_calls><invoke name="mcp__playwright__browser_navigate"></invoke></tool_calls>'
        calls = parse_tool_calls(
            text,
            tools,
            repair_missing=True,
            context="Open https://pmis.evn.com.vn/qlkt/tmsLogin.jsf now",
        )
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "mcp__playwright__browser_navigate")
        self.assertEqual(args["url"], "https://pmis.evn.com.vn/qlkt/tmsLogin.jsf")

    def test_tool_unavailable_text_is_not_a_tool_call(self):
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

        calls = parse_tool_calls("Tool Bash does not exists.", tools)

        self.assertIsNone(calls)

    def test_infers_recovery_tool_from_context_after_tool_miss_text(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "mcp__playwright__browser_navigate",
                "parameters": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {"url": {"type": "string"}},
                },
            },
        }]
        calls = infer_tool_calls_from_context(
            tools,
            "Load https://pmis.evn.com.vn/qlkt/tmsLogin.jsf and inspect it",
            "Tool Bash does not exists.",
        )
        args = json.loads(calls[0]["function"]["arguments"])

        self.assertEqual(calls[0]["function"]["name"], "mcp__playwright__browser_navigate")
        self.assertEqual(args["url"], "https://pmis.evn.com.vn/qlkt/tmsLogin.jsf")

    # -----------------------------------------------------------------
    # Prose-based recovery tests
    # -----------------------------------------------------------------

    def test_prose_recovery_bash_backtick_command(self):
        """Recover Bash tool call from 'I'll run `<command>`' prose."""
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
        calls = infer_tool_calls_from_context(
            tools,
            "List all Python files in /tmp",
            "I'll run `find /tmp -name '*.py'` to list all Python files.",
        )
        self.assertIsNotNone(calls)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(calls[0]["function"]["name"], "Bash")
        self.assertIn("find /tmp", args["command"])

    def test_prose_recovery_run_shell_command(self):
        """Recover run_shell_command from execute prose with backticks."""
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
        calls = infer_tool_calls_from_context(
            tools,
            "Check disk space",
            "Let me run `df -h` to check disk space.",
        )
        self.assertIsNotNone(calls)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(calls[0]["function"]["name"], "run_shell_command")
        self.assertEqual(args["command"], "df -h")

    def test_prose_recovery_read_file_absolute_path(self):
        """Recover read_file tool call from 'Let me read `/path/to/file`' prose."""
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
            },
        }]
        calls = infer_tool_calls_from_context(
            tools,
            "Show me /etc/hostname",
            "Let me read `/etc/hostname` to get the hostname.",
        )
        self.assertIsNotNone(calls)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(args["path"], "/etc/hostname")

    def test_prose_recovery_grep_search(self):
        """Recover grep_search from 'I'll search for `pattern` in `/path`' prose."""
        tools = [{
            "type": "function",
            "function": {
                "name": "grep_search",
                "parameters": {
                    "type": "object",
                    "required": ["pattern"],
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            },
        }]
        calls = infer_tool_calls_from_context(
            tools,
            "Find TODO comments in source",
            "I'll search for `TODO` in `/home/user/src`.",
        )
        self.assertIsNotNone(calls)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(calls[0]["function"]["name"], "grep_search")
        self.assertEqual(args["pattern"], "TODO")
        self.assertEqual(args["path"], "/home/user/src")

    def test_prose_recovery_does_not_trigger_on_ambiguous_text(self):
        """Do NOT recover when the model text is generic prose without patterns."""
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
        calls = infer_tool_calls_from_context(
            tools,
            "What time is it?",
            "The current time is approximately 10:30 AM.",
        )
        # Should not fabricate a Bash call from generic prose
        self.assertIsNone(calls)

    def test_url_in_final_answer_does_not_trigger_browser_recovery(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "mcp__playwright__browser_navigate",
                "parameters": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {"url": {"type": "string"}},
                },
            },
        }]
        calls = infer_tool_calls_from_context(
            tools,
            "Evaluate the macro malware report.",
            (
                "Đã đọc báo cáo. Kỹ thuật template injection dùng ví dụ "
                "http://update/Doc1.dotm nhưng hiện nay bị MOTW và Office policy "
                "chặn mạnh hơn, nên tính khả thi thấp."
            ),
        )

        self.assertIsNone(calls)

    def test_audit_text_with_past_navigation_does_not_trigger_browser_recovery(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "mcp__playwright__browser_navigate",
                "parameters": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {"url": {"type": "string"}},
                },
            },
        }]
        calls = infer_tool_calls_from_context(
            tools,
            "Audit notes/ops.md and do not open http://update/Doc1.dotm.",
            (
                "## Audit Report\n\n"
                "Trong session trước, assistant đã vi phạm khi gọi "
                "`mcp__playwright__browser_navigate` đến `http://update/Doc1.dotm` "
                "dù user đã cấm rõ ràng."
            ),
        )

        self.assertIsNone(calls)

    def test_url_in_prose_without_tool_miss_does_not_trigger_recovery(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "web_fetch",
                "parameters": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                },
            },
        }]
        calls = infer_tool_calls_from_context(
            tools,
            "Fetch https://example.com/data",
            "I'll run `curl https://example.com/data`.",
        )
        self.assertIsNone(calls)


if __name__ == "__main__":
    unittest.main()
