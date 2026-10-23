"""Test the LiteLLM adapter offline and the HTML review page built from its output."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import Mock, patch

from hpdv2.api import LiteLLMClient, CallError
from hpdv2.review import build_review
from hpdv2.storage import atomic_json, prompt_id
from hpdv2.dataset import SCHEMA


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.module = ModuleType("litellm")
        self.module.completion = Mock(
            return_value=NS(
                choices=[NS(message=NS(content="[]"), finish_reason="stop")],
                usage=NS(model_dump=lambda **_: {"prompt_tokens": 12}),
                model="resolved-model",
            )
        )
        self.config = dict(
            backend="proxy",
            model="anthropic/test",
            api_base="https://example.invalid",
            cache_system=True,
            timeout=30,
            max_tokens=100,
        )

    def test_proxy_payload_and_usage(self):
        with patch.dict("sys.modules", litellm=self.module):
            reply = LiteLLMClient(self.config, "test-secret")([], "policy")
        kwargs = self.module.completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "litellm_proxy/anthropic/test")
        self.assertEqual(kwargs["api_base"], "https://example.invalid")
        self.assertEqual(kwargs["num_retries"], 0)
        self.assertEqual(kwargs["timeout"], 30)
        self.assertIn("cache_control", kwargs["messages"][0]["content"][0])
        self.assertEqual(reply.usage["prompt_tokens"], 12)

    def test_direct_does_not_send_proxy_endpoint(self):
        with patch.dict("sys.modules", litellm=self.module):
            LiteLLMClient({**self.config, "backend": "anthropic", "cache_system": False}, "test")(
                [], "policy"
            )
        kwargs = self.module.completion.call_args.kwargs
        self.assertNotIn("api_base", kwargs)
        self.assertEqual(kwargs["model"], "anthropic/test")
        self.assertEqual(kwargs["messages"][0]["content"], "policy")

    def test_reasoning_effort_forwarded_without_temperature(self):
        with patch.dict("sys.modules", litellm=self.module):
            LiteLLMClient(
                {**self.config, "cache_system": False, "reasoning_effort": "low"}, "test"
            )([], "policy")
        kwargs = self.module.completion.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["max_tokens"], 100)

    def test_error_redaction_and_retry_classification(self):
        for status, retryable in [(401, False), (400, False), (429, True), (500, True)]:
            error = Exception("contains test-secret")
            error.status_code = status
            self.module.completion.side_effect = error
            with patch.dict("sys.modules", litellm=self.module), self.assertRaises(
                CallError
            ) as raised:
                LiteLLMClient(self.config, "test-secret")([], "policy")
            self.assertEqual(raised.exception.retryable, retryable)
            self.assertNotIn("test-secret", str(raised.exception))


class ReviewTests(unittest.TestCase):
    def test_html_escaped_and_sampling_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = '<script>alert("x")</script>'
            row = {
                "id": prompt_id(text),
                "prompt": text,
                "reason": "<unsafe>",
                "variation_axes": [{"dimension": "<clock time>", "values": ["<3:00>", "<9:00>"]}],
            }
            source = root / "summary.json"
            atomic_json(source, {"schema_version": SCHEMA, "kind": "filter_summary", "kept": [row]})
            build_review(source, root / "one.html")
            build_review(source, root / "two.html")
            html = (root / "one.html").read_text()
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("&lt;clock time&gt;", html)
            self.assertIn("&lt;3:00&gt;", html)
            self.assertEqual(html, (root / "two.html").read_text())


if __name__ == "__main__":
    unittest.main()
