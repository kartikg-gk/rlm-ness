"""Model backends."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx

Message = dict[str, str]


@dataclass(frozen=True)
class Provider:
    name: str
    endpoint: str
    env_var: str


OPENROUTER = Provider(
    "openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"
)
DEEPSEEK = Provider(
    "deepseek", "https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY"
)

PROVIDERS = {provider.name: provider for provider in (OPENROUTER, DEEPSEEK)}


class MissingApiKey(Exception):
    pass


@dataclass(frozen=True)
class Spend:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # None means the provider did not say, which is not the same as free.
    # Collapsing the two is what makes a spend limit quietly stop working.
    cost: float | None = None


class ModelClient(Protocol):
    def complete(self, messages: Sequence[Message], *, model: str) -> tuple[str, Spend]: ...


# Retrying anything else would repeat a request the provider has already
# rejected on its merits.
RETRIABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ChatClient:
    """Any OpenAI-compatible chat completions endpoint."""

    provider: Provider = OPENROUTER

    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: Provider | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        backoff: float = 0.5,
    ):
        self.provider = provider or type(self).provider
        key = api_key or os.environ.get(self.provider.env_var)
        if not key:
            raise MissingApiKey(f"set {self.provider.env_var}")
        self.api_key = key
        self.client = client or httpx.Client(timeout=timeout)
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = time.sleep

    def _post(self, messages: Sequence[Message], model: str) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = self.client.post(
                    self.provider.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": model, "messages": list(messages)},
                )
            except httpx.TransportError:
                if last:
                    raise
            else:
                if response.status_code not in RETRIABLE_STATUS or last:
                    response.raise_for_status()
                    return response
            self._sleep(self.backoff * 2**attempt)
        raise AssertionError("unreachable")

    def complete(self, messages: Sequence[Message], *, model: str) -> tuple[str, Spend]:
        response = self._post(messages, model)
        payload = response.json()
        raw = payload.get("usage") or {}
        reported = raw.get("cost")
        usage = Spend(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
            cost=float(reported) if reported is not None else None,
        )
        text = payload["choices"][0]["message"].get("content") or ""
        return text, usage


class OpenRouterClient(ChatClient):
    provider = OPENROUTER


class DeepSeekClient(ChatClient):
    provider = DEEPSEEK


def make_client(name: str, **options) -> ChatClient:
    return ChatClient(provider=PROVIDERS[name], **options)


class ScriptedClient:
    def __init__(self, replies: Sequence[str]):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, model: str) -> tuple[str, Spend]:
        self.calls.append(list(messages))
        assert self.replies, "ScriptedClient ran out of scripted replies"
        return self.replies.pop(0), Spend()
