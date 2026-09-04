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
    # Two more the provider may or may not break out. None means it did not,
    # for the same reason cost distinguishes silence from zero: a reader has
    # to be able to tell "no cache was hit" from "nobody counted".
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


def _sum_or_none(left, right):
    """Unknown on both sides stays unknown; known on either side is a total.

    A count nobody reported must not read as a measured zero — that is what
    turns a silent provider into a confident-looking nothing.
    """
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def combine(left: Spend, right: Spend) -> Spend:
    """Add two spends, keeping "nobody said" distinct from "zero"."""
    return Spend(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cost=_sum_or_none(left.cost, right.cost),
        cached_tokens=_sum_or_none(left.cached_tokens, right.cached_tokens),
        reasoning_tokens=_sum_or_none(left.reasoning_tokens, right.reasoning_tokens),
    )


def _detail(usage: dict, section: str, field: str) -> int | None:
    """Read a token count a provider may not break out at all.

    Two shapes are in the wild: nested under a details object, or flat
    alongside the totals. Absent in both is reported as absent rather than as
    zero, which would claim a measurement nobody made.
    """
    nested = usage.get(section)
    if isinstance(nested, dict) and nested.get(field) is not None:
        return int(nested[field])
    if usage.get(field) is not None:
        return int(usage[field])
    return None


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
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ):
        self.provider = provider or type(self).provider
        key = api_key or os.environ.get(self.provider.env_var)
        if not key:
            raise MissingApiKey(f"set {self.provider.env_var}")
        self.api_key = key
        self.client = client or httpx.Client(timeout=timeout)
        self.max_retries = max_retries
        self.backoff = backoff
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._reasoning_refused = False
        self._sleep = time.sleep

    def _body(self, messages: Sequence[Message], model: str) -> dict:
        """The request, carrying only the knobs that were actually set.

        An unset knob is left out rather than sent as a default, because not
        every endpoint accepts every field and a rejected request is worse
        than an unspecified one. Reasoning effort in particular is not
        universal, so it is dropped once a provider has refused it.
        """
        body: dict = {"model": model, "messages": list(messages)}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort and not self._reasoning_refused:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def _post(self, messages: Sequence[Message], model: str) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = self.client.post(
                    self.provider.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._body(messages, model),
                )
            except httpx.TimeoutException:
                # A provider that did not answer in the time allowed is not
                # likely to answer the identical question faster. One more try
                # covers a blip; three more just multiply the wait.
                if last or attempt >= 1:
                    raise
            except httpx.TransportError:
                if last:
                    raise
            else:
                if (
                    response.status_code == 400
                    and self.reasoning_effort
                    and not self._reasoning_refused
                ):
                    # Some endpoints reject the reasoning field outright. Drop
                    # it and try once more rather than failing the run over a
                    # setting that is an optimisation, not a requirement.
                    self._reasoning_refused = True
                    continue
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
            cached_tokens=_detail(raw, "prompt_tokens_details", "cached_tokens"),
            reasoning_tokens=_detail(
                raw, "completion_tokens_details", "reasoning_tokens"
            ),
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
