"""Tool-call parsing and OpenAI formatting helpers.

The primary model-facing format is DSML/XML, borrowed from ds2api's approach:

<|DSML|tool_calls>
  <|DSML|invoke name="read_file">
    <|DSML|parameter name="path"><![CDATA[/tmp/a.txt]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>

JSON tool_calls are kept as a migration fallback for the original Qwen proxy.
"""

from __future__ import annotations

import html
import json
import re
import uuid
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Any


DSML_TAGS = ("tool_calls", "invoke", "parameter")
CONFUSABLE_MAP = str.maketrans({
    "｜": "|",
    "＜": "<",
    "＞": ">",
})


def parse_tool_calls(
    text: str,
    tools: list[dict[str, Any]] | None = None,
    *,
    allow_incomplete: bool = True,
    repair_missing: bool = False,
    context: str = "",
) -> list[dict[str, Any]] | None:
    text = _normalize_markup_chars(text or "")
    parsed = _parse_xml_tool_calls(text, allow_incomplete=allow_incomplete)
    # _extract_from_broken_dsml is also a lenient/incomplete parse path:
    # only run it when allow_incomplete is True (i.e., after stream has ended).
    if not parsed and allow_incomplete:
        parsed = _extract_from_broken_dsml(text, tools)
    if not parsed:
        parsed = _parse_legacy_single_tool_call(text, tools)
    if not parsed:
        parsed = _parse_json_tool_calls(text)
    if not parsed:
        return None

    parsed = [_normalize_call_arguments(call) for call in parsed]
    normalized = _normalize_calls_for_schemas(parsed, tools)
    normalized = _validate_calls(normalized, tools, repair_missing=repair_missing, context=context)
    if not normalized:
        return None
    return [_format_openai_tool_call(call) for call in normalized]


def infer_tool_calls_from_context(
    tools: list[dict[str, Any]] | None,
    context: str,
    model_text: str = "",
) -> list[dict[str, Any]] | None:
    """Synthesize a conservative recovery tool call from user context.

    Some non-native tool models respond with prose such as "Tool X does not
    exist" instead of emitting the prompted DSML block. When the requested next
    action is unambiguous, recover by returning exactly one valid tool call.
    """
    tool_names = _tool_names(tools)
    if not tool_names:
        return None

    text = f"{context}\n{model_text}"
    lowered = text.lower()

    # Robust skill-loaded detection: check multiple markers that indicate
    # the Skill tool was already called in this conversation.
    _SKILL_LOADED_MARKERS = (
        "launching skill:",
        "base directory for this skill:",
        "successfully loaded skill",
        "loaded skill",
        "tool name: skill",         # From flatten_messages tool_result format
        'invoke name="skill"',      # From DSML in conversation history
        '"name": "skill"',          # From JSON tool call in history
    )
    skill_loaded = any(marker in lowered for marker in _SKILL_LOADED_MARKERS)

    skill_tool_name = _find_tool_name(tool_names, "skill")
    if not skill_loaded and skill_tool_name:
        skill_name = _infer_skill_name(text)
        if skill_name:
            return [_format_openai_tool_call({
                "name": skill_tool_name,
                "arguments": _skill_arguments_for_tools(tools, skill_name),
            })]

    url = _first_url(text)
    if url:
        if "mcp__playwright__browser_navigate" in tool_names:
            return [_format_openai_tool_call({
                "name": "mcp__playwright__browser_navigate",
                "arguments": {"url": url},
            })]
        if "WebFetch" in tool_names:
            return [_format_openai_tool_call({
                "name": "WebFetch",
                "arguments": {"url": url, "prompt": "Fetch this page and extract relevant links, scripts, and APK references."},
            })]
        if "web_fetch" in tool_names:
            return [_format_openai_tool_call({
                "name": "web_fetch",
                "arguments": {"url": url, "prompt": "Fetch this page and extract relevant links, scripts, and APK references."},
            })]

    # Prose-based recovery: when Qwen describes the action in text instead of DSML
    prose_call = _infer_tool_from_prose(model_text, tool_names)
    if prose_call:
        return [_format_openai_tool_call(prose_call)]

    return None


def _format_openai_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args or {}, ensure_ascii=False)
    return {
        "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": args,
        },
    }


def _parse_xml_tool_calls(text: str, *, allow_incomplete: bool = True) -> list[dict[str, Any]]:
    candidate = _extract_xml_tool_calls(text, allow_incomplete=allow_incomplete)
    if not candidate:
        return []
    normalized = _normalize_dsml_markup(candidate)
    try:
        root = ET.fromstring(normalized)
    except ET.ParseError:
        return []
    if _local_name(root.tag) != "tool_calls":
        return []

    calls: list[dict[str, Any]] = []
    for invoke in root:
        if _local_name(invoke.tag) != "invoke":
            continue
        name = html.unescape((invoke.attrib.get("name") or "").strip())
        if not name:
            continue
        args: dict[str, Any] = {}
        for param in invoke:
            if _local_name(param.tag) != "parameter":
                continue
            pname = html.unescape((param.attrib.get("name") or "").strip())
            if not pname:
                continue
            args[pname] = _parse_parameter_value(param)
        calls.append({"name": name, "arguments": args})
    return calls


def _extract_xml_tool_calls(text: str, *, allow_incomplete: bool = True) -> str | None:
    normalized = _normalize_dsml_markup(text or "")
    match = re.search(r"<tool_calls\b", normalized, re.IGNORECASE)
    if not match:
        return None
    start = match.start()
    end_match = re.search(r"</tool_calls\s*>", normalized[start:], re.IGNORECASE)
    if end_match:
        return normalized[start : start + end_match.end()]
    if not allow_incomplete:
        return None
    # Truncated DSML: Qwen stream was cut short. Repair and retry.
    return _repair_truncated_xml(normalized[start:])


def _repair_truncated_xml(fragment: str) -> str | None:
    """Repair a truncated DSML/XML tool_calls block by closing open tags.

    Qwen sometimes truncates its response (token limit, timeout), leaving
    an incomplete DSML block. We close any open tags so the XML parser
    can extract whatever parameters were completed.
    """
    # Must have a completed invoke or parameter before fabricating close tags.
    # This covers zero-argument tools while avoiding "<invoke name=...>" alone.
    lowered = fragment.lower()
    if "</parameter>" not in lowered and "</invoke>" not in lowered:
        return None
    # Remove trailing incomplete tag (e.g., "<" or "<|DSM" or "<invoke nam")
    repaired = re.sub(r"<[^>]*$", "", fragment)
    # Close open CDATA sections
    if repaired.count("<![CDATA[") > repaired.count("]]>"):
        repaired += "]]>"
    # Close open parameter tags
    open_params = repaired.count("<parameter") - repaired.count("</parameter")
    for _ in range(max(0, open_params)):
        repaired += "</parameter>"
    # Close open invoke tags
    open_invokes = repaired.count("<invoke") - repaired.count("</invoke")
    for _ in range(max(0, open_invokes)):
        repaired += "</invoke>"
    # Close tool_calls tag
    if "</tool_calls>" not in repaired.lower():
        repaired += "</tool_calls>"
    return repaired



def _normalize_dsml_markup(text: str) -> str:
    out = _normalize_markup_chars(text or "")
    for tag in DSML_TAGS:
        # Closing tags: </|DSML|tag>, </|/DSML|tag>, </||DSML|tag>, etc.
        out = re.sub(
            rf"<\s*/\s*[\|｜]*\s*/?\s*DSML\s*[\|｜]\s*{tag}\s*>",
            f"</{tag}>",
            out,
            flags=re.IGNORECASE,
        )
        # Opening tags: <|DSML|tag>, <||DSML|tag>, <|||DSML|tag>, etc.
        out = re.sub(
            rf"<\s*[\|｜]+\s*DSML\s*[\|｜]\s*{tag}(\s[^>]*)?",
            lambda m: f"<{tag}{m.group(1) or ''}",
            out,
            flags=re.IGNORECASE,
        )
    # Strip garbage tokens Qwen sometimes hallucinates: <||>, <|>, etc.
    out = re.sub(r'<\|+\s*>', '', out)
    return out


def _extract_from_broken_dsml(
    text: str,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Regex-based extraction from malformed DSML when XML parse fails.

    When Qwen emits garbled DSML that standard XML parsing cannot handle,
    this function falls back to regex-based extraction of invoke names and
    CDATA parameter values.
    """
    normalized = _normalize_dsml_markup(text or "")
    name_matches = list(re.finditer(r'invoke\s+name="([^"]+)"', normalized, re.IGNORECASE))
    if not name_matches:
        return []

    calls: list[dict[str, Any]] = []
    for i, name_match in enumerate(name_matches):
        name = name_match.group(1)
        start = name_match.end()
        end = name_matches[i + 1].start() if i + 1 < len(name_matches) else len(normalized)
        block = normalized[start:end]

        args = _extract_parameters_from_broken_block(block)
        if name and args:
            calls.append({"name": name, "arguments": args})

    return calls


def _extract_parameters_from_broken_block(block: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for param_match in re.finditer(
        r'<\s*parameter(?:\s+name\s*=\s*["\']?([^"\'\s>]+)["\']?|\s*=\s*["\']?([^"\'\s>]+)["\']?)[^>]*>(.*?)(?:</\s*parameter\s*>|$)',
        block,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        pname = (param_match.group(1) or param_match.group(2) or "").strip()
        if not pname:
            continue
        raw = param_match.group(3) or ""
        cdata = re.search(r'<!\[CDATA\[(.*?)(?:\]\]>|$)', raw, flags=re.DOTALL)
        if cdata:
            value = cdata.group(1)
        else:
            value = html.unescape(raw).strip()
        if value:
            args[pname] = value
    return args


def _normalize_markup_chars(text: str) -> str:
    if not text:
        return ""
    return text.translate(CONFUSABLE_MAP)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _parse_parameter_value(param: ET.Element) -> Any:
    children = list(param)
    if children:
        return _element_to_value(param)
    raw = html.unescape(param.text or "")
    stripped = raw.strip()
    if stripped == "":
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


def _element_to_value(el: ET.Element) -> Any:
    children = list(el)
    if not children:
        return html.unescape(el.text or "")
    if all(_local_name(child.tag) == "item" for child in children):
        return [_element_to_value(child) for child in children]
    obj: dict[str, Any] = {}
    for child in children:
        key = _local_name(child.tag)
        value = _element_to_value(child)
        if key in obj:
            if not isinstance(obj[key], list):
                obj[key] = [obj[key]]
            obj[key].append(value)
        else:
            obj[key] = value
    return obj


def _parse_json_tool_calls(text: str) -> list[dict[str, Any]]:
    for candidate in _json_candidates(text or ""):
        obj = _try_parse_json(candidate)
        if not isinstance(obj, dict):
            continue
        raw_calls = obj.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        calls: list[dict[str, Any]] = []
        for tc in raw_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = (fn.get("name") or tc.get("name") or "").strip()
            if not name:
                continue
            args = fn.get("arguments", tc.get("arguments", {}))
            if isinstance(args, str):
                args_obj = _try_parse_json(args)
                args = args_obj if args_obj is not None else args
            call_id = tc.get("id")
            if not call_id and isinstance(args, dict):
                nested_id = args.get("id")
                if isinstance(nested_id, str) and nested_id.startswith("call_"):
                    call_id = nested_id
                    args = dict(args)
                    args.pop("id", None)
            calls.append({"id": call_id, "name": name, "arguments": args})
        if calls:
            return calls
    return []


def _parse_legacy_single_tool_call(
    text: str,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for body in re.findall(r"<\s*[\|｜]?\s*tool_call\s*[\|｜]?\s*>(.*?)</\s*[\|｜]?\s*tool_call\s*[\|｜]?\s*>", text or "", flags=re.DOTALL | re.IGNORECASE):
        obj = _try_parse_json(html.unescape(body.strip()))
        call = _coerce_tool_call_object(obj, tools)
        if call:
            calls.append(call)
    return calls


def _coerce_tool_call_object(
    obj: Any,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    tool_names = _tool_names(tools)
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
    raw_id = str(obj.get("id") or "").strip()
    name = str(fn.get("name") or obj.get("name") or obj.get("tool") or obj.get("tool_name") or "").strip()
    call_id = raw_id
    if not name and raw_id:
        name = raw_id
        call_id = ""
    if tool_names and name not in tool_names:
        lower_index = {item.lower(): item for item in tool_names}
        name = lower_index.get(name.lower(), name)
    if not name:
        return None
    args = fn.get("arguments", obj.get("arguments", obj.get("input", {})))
    if isinstance(args, str):
        parsed_args = _try_parse_json(args)
        args = parsed_args if parsed_args is not None else args
    return {"id": call_id or None, "name": name, "arguments": args}


def _json_candidates(text: str) -> list[str]:
    candidates = re.findall(r"```(?:json)?\s*(\{.+?\})\s*```", text, flags=re.DOTALL)
    match = re.search(r'(\{"tool_calls"\s*:\s*\[.+\]\s*\})', text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    return candidates


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        repaired = _repair_json_closers(text)
        if repaired and repaired != text:
            try:
                return json.loads(repaired)
            except (json.JSONDecodeError, TypeError):
                return None
        return None


def _repair_json_closers(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip().startswith(("{", "[")):
        return None

    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        out.append(ch)
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            while stack and stack[-1] != ch:
                out.insert(len(out) - 1, stack.pop())
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        return None
    while stack:
        out.append(stack.pop())
    return "".join(out)


def _normalize_calls_for_schemas(
    calls: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    schemas = _schema_index(tools)
    if not schemas:
        return calls
    out = deepcopy(calls)
    for call in out:
        schema = schemas.get(call["name"].lower())
        args = call.get("arguments")
        if schema and isinstance(args, dict):
            call["arguments"] = _normalize_value(args, schema)
    return out


def _validate_calls(
    calls: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    repair_missing: bool = False,
    context: str = "",
) -> list[dict[str, Any]]:
    if not calls:
        return []
    tool_names = _tool_names(tools)
    required = _required_fields_index(tools)
    out: list[dict[str, Any]] = []

    for call in calls:
        name = str(call.get("name") or "").strip()
        if not name:
            continue
        if tool_names and name not in tool_names:
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            continue
        required_fields = required.get(name.lower(), [])
        if required_fields:
            args = _repair_argument_aliases(name, args, required_fields)
            call["arguments"] = args
        if required_fields and repair_missing:
            args = _fill_inferred_required_args(name, args, required_fields, context)
            call["arguments"] = args
        if any(_is_missing_required(args.get(field)) for field in required_fields):
            continue
        out.append(call)
    return out


def _repair_argument_aliases(
    tool_name: str,
    args: dict[str, Any],
    required_fields: list[str],
) -> dict[str, Any]:
    """Repair emitted argument names without inventing values.

    Qwen sometimes serializes a real tool call with a wrong-but-obvious parameter
    label, e.g. Bash `<parameter=code>` although Claude Code requires
    `command`. This keeps strict required-field validation while preserving the
    value the model actually emitted.
    """
    if not args:
        return args
    repaired = dict(args)
    required_set = {field.lower(): field for field in required_fields}

    aliases = {
        "command": ("code", "cmd", "script", "shell", "bash"),
        "file_path": ("path", "filepath", "filename", "file"),
        "path": ("file_path", "filepath", "filename", "file"),
        "url": ("uri", "link", "href"),
        "query": ("queries", "q", "search", "search_query"),
        "queries": ("query", "q", "search", "search_query"),
    }
    if tool_name.lower() == "skill":
        aliases["skill"] = ("name", "skill_name", "id")

    lowered_args = {str(key).lower(): key for key in repaired}
    for canonical_lower, alias_names in aliases.items():
        canonical = required_set.get(canonical_lower)
        if not canonical or not _is_missing_required(repaired.get(canonical)):
            continue
        for alias in alias_names:
            actual = lowered_args.get(alias)
            if actual is not None and not _is_missing_required(repaired.get(actual)):
                repaired[canonical] = _coerce_alias_value(canonical, repaired[actual])
                if actual != canonical:
                    repaired.pop(actual, None)
                break
    return repaired


def _coerce_alias_value(canonical: str, value: Any) -> Any:
    if canonical == "query":
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if isinstance(value, str):
            parsed = _try_parse_json(value.strip())
            if isinstance(parsed, list):
                return str(parsed[0]) if parsed else ""
        return value
    if canonical == "queries":
        if isinstance(value, str):
            parsed = _try_parse_json(value.strip())
            if isinstance(parsed, list):
                return parsed
            return [value]
    return value


def _fill_inferred_required_args(
    name: str,
    args: dict[str, Any],
    required_fields: list[str],
    context: str,
) -> dict[str, Any]:
    """Repair common empty tool calls when the user context is unambiguous."""
    lowered_name = name.lower()

    if "url" in required_fields and _is_missing_required(args.get("url")):
        url = _first_url(context)
        if url and ("navigate" in lowered_name or "webfetch" in lowered_name or "web_fetch" in lowered_name):
            repaired = dict(args)
            repaired["url"] = url
            if "prompt" in required_fields and _is_missing_required(repaired.get("prompt")):
                repaired["prompt"] = "Fetch this page and extract relevant links, scripts, and APK references."
            return repaired

    if lowered_name != "skill":
        return args

    inferred = _infer_skill_name(context)
    if not inferred:
        return args

    repaired = dict(args)
    for field in required_fields:
        if _is_missing_required(repaired.get(field)):
            repaired[field] = inferred
    return repaired


def _skill_arguments_for_tools(tools: list[dict[str, Any]] | None, skill_name: str) -> dict[str, Any]:
    required = _required_fields_index(tools).get("skill", [])
    schemas = _schema_index(tools)
    props = (schemas.get("skill") or {}).get("properties") or {}
    for key in ("skill", "name", "skill_name", "id"):
        if key in props or key in required:
            return {key: skill_name}
    if required:
        return {required[0]: skill_name}
    return {"skill": skill_name}


def _find_tool_name(tool_names: set[str], wanted: str) -> str | None:
    wanted = wanted.lower()
    for name in tool_names:
        if name.lower() == wanted:
            return name
    return None


def _infer_skill_name(context: str) -> str | None:
    lowered = (context or "").lower()
    if "pentest-assistant-reasoning" in lowered:
        return "pentest-assistant-reasoning"
    if "pentest" in lowered and "assistant" in lowered and ("reasoning" in lowered or "reassioning" in lowered):
        return "pentest-assistant-reasoning"
    if "pentet" in lowered and "assistant" in lowered and ("reasoning" in lowered or "reassioning" in lowered):
        return "pentest-assistant-reasoning"
    return None


# ---------------------------------------------------------------------------
# Prose-based tool inference patterns
# ---------------------------------------------------------------------------
# Each entry: (compiled regex on model_text, candidate tool names, argument builder)
# The regex must capture the key argument (command, path, etc.).

_PROSE_BASH_RE = re.compile(
    r"(?:I(?:'ll| will| want to| need to| can)?|Let me|Let's)"
    r"\s+(?:run|execute|use|try|call|invoke)\s+"
    r"(?:the\s+(?:following\s+)?(?:command|shell command|bash command)[:\s]*)?`([^`]+)`",
    re.IGNORECASE,
)

_PROSE_BASH_CODEBLOCK_RE = re.compile(
    r"(?:I(?:'ll| will)?|Let me|Let's)\s+(?:run|execute)\s+(?:this|the following)[^`]*"
    r"```(?:bash|sh|shell)?\s*\n([^\n]+)\n```",
    re.IGNORECASE | re.DOTALL,
)

_PROSE_READ_FILE_RE = re.compile(
    r"(?:I(?:'ll| will)?|Let me|Let's)\s+(?:read|open|view|check|look at|examine|inspect)\s+"
    r"(?:the\s+(?:file|content(?:s)?\s+of)\s+)?`(/[^`]+)`",
    re.IGNORECASE,
)

_PROSE_LIST_DIR_RE = re.compile(
    r"(?:I(?:'ll| will)?|Let me|Let's)\s+(?:list|check|view|look at)\s+"
    r"(?:the\s+)?(?:files|contents|directory)\s+"
    r"(?:in|of|at)\s+`(/[^`]+)`",
    re.IGNORECASE,
)

_PROSE_GREP_RE = re.compile(
    r"(?:I(?:'ll| will)?|Let me|Let's)\s+(?:search|grep|find|look)\s+"
    r"(?:for\s+)?`([^`]+)`\s+(?:in|inside|within)\s+`([^`]+)`",
    re.IGNORECASE,
)

# Map of alternative tool names (lowercase) -> preferred name
_BASH_TOOL_ALIASES = {"bash", "run_shell_command", "shell", "execute_command"}
_READ_TOOL_ALIASES = {"read_file", "readfile", "file_read"}
_LIST_TOOL_ALIASES = {"list_directory", "listdirectory", "list_dir"}
_GREP_TOOL_ALIASES = {"grep_search", "grep", "search_files"}


def _infer_tool_from_prose(
    model_text: str,
    tool_names: set[str],
) -> dict[str, Any] | None:
    """Extract tool call from model prose when it describes what it would do.

    Only triggers on unambiguous patterns like:
      "I'll run `ls -la`"
      "Let me read `/etc/hostname`"
      "I'll search for `pattern` in `/path`"

    Returns a single tool call dict or None if no confident match.
    """
    if not model_text or not tool_names:
        return None

    text = model_text.strip()

    # --- Bash / run_shell_command ---
    bash_name = _find_tool_name_fuzzy(tool_names, _BASH_TOOL_ALIASES)
    if bash_name:
        match = _PROSE_BASH_RE.search(text) or _PROSE_BASH_CODEBLOCK_RE.search(text)
        if match:
            command = match.group(1).strip()
            if command and len(command) > 1:
                return {"name": bash_name, "arguments": {"command": command}}

    # --- read_file ---
    read_name = _find_tool_name_fuzzy(tool_names, _READ_TOOL_ALIASES)
    if read_name:
        match = _PROSE_READ_FILE_RE.search(text)
        if match:
            path = match.group(1).strip()
            if path:
                return {"name": read_name, "arguments": {"path": path}}

    # --- list_directory ---
    list_name = _find_tool_name_fuzzy(tool_names, _LIST_TOOL_ALIASES)
    if list_name:
        match = _PROSE_LIST_DIR_RE.search(text)
        if match:
            path = match.group(1).strip()
            if path:
                return {"name": list_name, "arguments": {"path": path}}

    # --- grep_search ---
    grep_name = _find_tool_name_fuzzy(tool_names, _GREP_TOOL_ALIASES)
    if grep_name:
        match = _PROSE_GREP_RE.search(text)
        if match:
            pattern = match.group(1).strip()
            search_path = match.group(2).strip()
            if pattern:
                return {"name": grep_name, "arguments": {"pattern": pattern, "path": search_path}}

    return None


def _find_tool_name_fuzzy(tool_names: set[str], aliases: set[str]) -> str | None:
    """Find the first tool name that matches any of the given aliases (case-insensitive)."""
    lower_map = {name.lower(): name for name in tool_names}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _first_url(context: str) -> str | None:
    match = re.search(r"https?://[^\s<>'\"`]+", context or "")
    if not match:
        return None
    return match.group(0).rstrip(").,;]")


def _is_missing_required(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _normalize_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments")
    if isinstance(args, dict):
        call = dict(call)
        call["arguments"] = {key: _normalize_argument_value(key, value) for key, value in args.items()}
    return call


def _normalize_argument_value(key: str, value: Any) -> Any:
    if isinstance(value, str) and key.lower() in {"url", "uri", "link", "href"}:
        return _clean_url_value(value)
    if isinstance(value, dict):
        return {child_key: _normalize_argument_value(child_key, child) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_argument_value(key, item) for item in value]
    return value


def _clean_url_value(value: str) -> str:
    stripped = value.strip()
    markdown = re.fullmatch(r"\[[^\]]+\]\((https?://[^)\s]+)\)", stripped)
    if markdown:
        return markdown.group(1)
    leading = re.match(r"(https?://[^\s()]+)(?:\s+\(https?://[^)]+\))?$", stripped)
    if leading:
        return leading.group(1)
    return value


def _schema_index(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = (fn.get("name") or "").strip()
        schema = fn.get("parameters") or fn.get("input_schema") or fn.get("inputSchema")
        if name and isinstance(schema, dict):
            out[name.lower()] = schema
    return out


def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = (fn.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _required_fields_index(tools: list[dict[str, Any]] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = (fn.get("name") or "").strip()
        schema = fn.get("parameters") or fn.get("input_schema") or fn.get("inputSchema")
        if not name or not isinstance(schema, dict):
            continue
        required = schema.get("required")
        if isinstance(required, list):
            out[name.lower()] = [item for item in required if isinstance(item, str)]
    return out


def _normalize_value(value: Any, schema: dict[str, Any]) -> Any:
    if _schema_is_string(schema):
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)
    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        return {
            key: _normalize_value(child, props[key]) if isinstance(props.get(key), dict) else child
            for key, child in value.items()
        }
    if schema_type == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        item_schema = schema["items"]
        return [_normalize_value(item, item_schema) for item in value]
    return value


def _schema_is_string(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type.lower() == "string"
    if isinstance(schema_type, list):
        return schema_type and all(str(item).lower() in {"string", "null"} for item in schema_type)
    return False
