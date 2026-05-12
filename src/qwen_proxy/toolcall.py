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


def parse_tool_calls(text: str, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]] | None:
    text = _normalize_markup_chars(text or "")
    parsed = _parse_xml_tool_calls(text)
    if not parsed:
        parsed = _parse_legacy_single_tool_call(text, tools)
    if not parsed:
        parsed = _parse_json_tool_calls(text)
    if not parsed:
        return None

    parsed = [_normalize_call_arguments(call) for call in parsed]
    normalized = _normalize_calls_for_schemas(parsed, tools)
    normalized = _validate_calls(normalized, tools)
    if not normalized:
        return None
    return [_format_openai_tool_call(call) for call in normalized]


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


def _parse_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    candidate = _extract_xml_tool_calls(text)
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


def _extract_xml_tool_calls(text: str) -> str | None:
    normalized = _normalize_dsml_markup(text or "")
    match = re.search(r"<tool_calls\b", normalized, re.IGNORECASE)
    if not match:
        return None
    start = match.start()
    end_match = re.search(r"</tool_calls\s*>", normalized[start:], re.IGNORECASE)
    if not end_match:
        return None
    end = start + end_match.end()
    return normalized[start:end]


def _normalize_dsml_markup(text: str) -> str:
    out = _normalize_markup_chars(text or "")
    for tag in DSML_TAGS:
        out = re.sub(
            rf"<\s*[\|｜]?\s*DSML\s*[\|｜]\s*{tag}(\s[^>]*)?>",
            lambda m: f"<{tag}{m.group(1) or ''}>",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            rf"</\s*[\|｜]?\s*DSML\s*[\|｜]\s*{tag}\s*>",
            f"</{tag}>",
            out,
            flags=re.IGNORECASE,
        )
    return out


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
            calls.append({"id": tc.get("id"), "name": name, "arguments": args})
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
        return None


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


def _validate_calls(calls: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
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
        if any(_is_missing_required(args.get(field)) for field in required_fields):
            continue
        out.append(call)
    return out


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
