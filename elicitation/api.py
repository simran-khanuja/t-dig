"""Explicit OpenAI-compatible gateway boundary; no silent provider fallbacks."""

from dataclasses import dataclass


@dataclass
class Reply:
    text: str | None
    finish_reason: str
    usage: dict
    model: str
    refusal: str | None = None


class ProviderError(Exception):
    def __init__(self, name, status):
        super().__init__(f'{name} (HTTP {status})')
        self.status = status


class Gateway:
    def __init__(self, config, api_key):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key, base_url=config['api_base'], max_retries=0, timeout=config['timeout']
        )
        self.config = config

    def __call__(self, payload):
        try:
            result = self.client.chat.completions.create(**payload)
        except Exception as error:
            # Never persist exception bodies, URLs, headers or credentials.
            raise ProviderError(type(error).__name__, getattr(error, 'status_code', None)) from None
        choice = result.choices[0]
        return Reply(
            choice.message.content,
            choice.finish_reason,
            result.usage.model_dump() if result.usage else {},
            result.model,
            getattr(choice.message, 'refusal', None),
        )
