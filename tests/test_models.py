import unittest

from qwen_proxy.server import (
    DEFAULT_MODEL,
    DEFAULT_TOOL_MODEL,
    RETRY_ON_FAIL,
    TOOL_MODEL,
    TOOL_RECOVERY,
    MODELS_LIST,
    resolve_model,
    select_upstream_model,
)


class ModelRegistryTests(unittest.TestCase):
    def test_defaults_to_qwen_36_plus_for_tool_compatibility(self):
        self.assertEqual(DEFAULT_MODEL, "qwen3.6-plus")
        self.assertEqual(resolve_model(""), ("qwen3.6-plus", "auto"))

    def test_loop_prone_recovery_features_are_opt_in(self):
        self.assertFalse(RETRY_ON_FAIL)
        self.assertFalse(TOOL_RECOVERY)

    def test_tool_requests_default_to_stable_tool_model(self):
        self.assertEqual(DEFAULT_TOOL_MODEL, "qwen3.6-max-preview")
        self.assertEqual(TOOL_MODEL, "qwen3.6-max-preview")
        self.assertEqual(select_upstream_model("qwen3.7-max", has_tools=True), "qwen3.6-max-preview")
        self.assertEqual(select_upstream_model("qwen3.7-max", has_tools=False), "qwen3.7-max")

    def test_resolves_thinking_and_fast_suffixes(self):
        self.assertEqual(resolve_model("qwen3.7-max-thinking"), ("qwen3.7-max", "thinking"))
        self.assertEqual(resolve_model("qwen3.7-max-fast"), ("qwen3.7-max", "fast"))

    def test_resolves_legacy_aliases(self):
        self.assertEqual(resolve_model("qwen3.7-max-preview"), ("qwen-latest-series-invite-beta-v24", "auto"))
        self.assertEqual(resolve_model("qwen3.7-plus-preview"), ("qwen-latest-series-invite-beta-v16", "auto"))
        self.assertEqual(resolve_model("qwen3.6-max"), ("qwen3.6-max-preview", "auto"))
        self.assertEqual(resolve_model("qwen3.5-max-preview"), ("qwen3.5-max-2026-03-08", "auto"))

    def test_strips_provider_prefixes(self):
        self.assertEqual(resolve_model("ac-qwen-proxy/qwen3.7-max-thinking"), ("qwen3.7-max", "thinking"))
        self.assertEqual(resolve_model("openai/qwen3.6-plus-fast"), ("qwen3.6-plus", "fast"))

    def test_models_endpoint_list_includes_all_modes(self):
        ids = {item["id"] for item in MODELS_LIST}
        for model_id in (
            "qwen3.7-max",
            "qwen3.7-max-thinking",
            "qwen3.7-max-fast",
            "qwen3.7-max-preview",
            "qwen3.7-plus-preview",
            "qwen3.6-plus",
            "qwen3.6-max-preview",
            "qwen3.6-max-preview-thinking",
            "qwen3.6-27b",
            "qwen3-coder-plus",
            "qwen3-max",
        ):
            self.assertIn(model_id, ids)

        self.assertNotIn("qwen3.7-plus", ids)


if __name__ == "__main__":
    unittest.main()
