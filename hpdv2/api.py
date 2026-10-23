"""Optional LiteLLM boundary. Endpoint choice is explicit; SDK retries disabled."""

from __future__ import annotations

from dataclasses import dataclass, field
import json


@dataclass
class Reply:
    text: str
    finish_reason: str
    usage: dict = field(default_factory=dict)
    model: str | None = None


class CallError(Exception):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class LiteLLMClient:
    def __init__(self, config, api_key):
        from litellm import completion

        self.completion = completion
        self.config = config
        self.api_key = api_key

    def __call__(self, batch, system):
        cfg = self.config
        content = system
        if cfg["cache_system"]:
            content = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        model = cfg["model"]
        if cfg["backend"] == "proxy":
            model = "litellm_proxy/" + model
        elif not model.startswith("anthropic/"):
            model = "anthropic/" + model
        kwargs = dict(
            model=model,
            api_key=self.api_key,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=cfg["max_tokens"],
            timeout=cfg["timeout"],
            num_retries=0,
        )
        if cfg.get("reasoning_effort") is not None:
            kwargs["reasoning_effort"] = cfg["reasoning_effort"]
            # Reasoning models may reject temperature=0. Use the model default.
            kwargs.pop("temperature")
        # Never send a direct Anthropic key to a proxy through an implicit default.
        if cfg["backend"] == "proxy":
            kwargs["api_base"] = cfg["api_base"]
        try:
            response = self.completion(**kwargs)
        except Exception as error:
            status = getattr(error, "status_code", None)
            retryable = (
                status in {408, 409, 429}
                or isinstance(status, int)
                and status >= 500
                or type(error).__name__ in {"Timeout", "APIConnectionError"}
            )
            # SDK messages may contain headers/keys; persist only type and status.
            raise CallError(f"{type(error).__name__} (HTTP {status})", retryable) from None
        try:
            choice = response.choices[0]
            usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
            return Reply(choice.message.content, choice.finish_reason, usage, response.model)
        except (AttributeError, IndexError, TypeError):
            raise CallError("Malformed provider response", retryable=True) from None
