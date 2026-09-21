"""Model backends."""

from __future__ import annotations

import json
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
ANTHROPIC = Provider(
    "anthropic", "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY"
)

PROVIDERS = {
    provider.name: provider for provider in (OPENROUTER, DEEPSEEK, ANTHROPIC)
}


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


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _cost(usage: dict) -> float | None:
    """What the call cost, including when the key brings its own provider.

    With a bring-your-own-key account the top-level cost is only the router's
    fee, often zero, while the inference is billed upstream and reported
    separately. Reading the fee alone lets a spend limit watch almost nothing
    while the real bill grows.
    """
    top = _number(usage.get("cost"))
    details = usage.get("cost_details")
    upstream = _number(details.get("upstream_inference_cost")) if isinstance(details, dict) else None
    if usage.get("is_byok") is True and upstream is not None:
        return upstream + (top if top and top > 0 else 0.0)
    if top is not None and top > 0:
        return top
    if upstream is not None and upstream > 0:
        return upstream
    return top


#: Marks a call the model made in its own tool-call format rather than in code.
NATIVE_CALL = "[tool call, not run]"


def native_call(name: str, arguments) -> str:
    """A tool call the model emitted, written out where the reply text goes.

    Some models answer with their own tool-call format even when no tools are
    offered, and the provider moves that call out of the reply into a field of
    its own. Reading only the reply then sees nothing at all, and the model is
    told it sent nothing — when it had in fact tried to act. Kept as text, the
    call stays in its transcript, and the loop can say what went wrong.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return f"\n\n{NATIVE_CALL} {name}({arguments[:300]})"
    if isinstance(arguments, dict):
        shown = ", ".join(f"{key}={value!r}"[:200] for key, value in arguments.items())
    else:
        shown = repr(arguments)[:300]
    return f"\n\n{NATIVE_CALL} {name}({shown})"


def _reasoning_text(message: dict) -> str | None:
    """The model's thinking, where the provider breaks it out.

    Two shapes are in the wild: a plain string, or a list of blocks each
    carrying its own text. Neither is universal and many providers send
    nothing at all, so absent stays absent rather than becoming an empty
    string -- "the model did not think aloud" and "nobody recorded it" are
    different facts about a run.
    """
    raw = message.get("reasoning")
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, list):
        parts = [
            block.get("text") or block.get("thinking") or ""
            for block in raw
            if isinstance(block, dict)
        ]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return None


class ModelClient(Protocol):
    def complete(
        self, messages: Sequence[Message], *, model: str
    ) -> tuple[str, Spend] | tuple[str, Spend, str | None]: ...


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
        # A minute, matching the config default. A caller that forgets to
        # pass one should still get a bound worth having: threading a
        # setting through every caller does not help the caller that was
        # written before the setting existed.
        timeout: float = 60.0,
        backoff: float = 0.5,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        # A bound on a whole reply, where `timeout` bounds one read of it. A
        # reply that keeps sending a few bytes at a time resets the read
        # timeout forever, and a fan-out waits on whichever child is stuck.
        deadline: float | None = 600.0,
        # However long a provider says to wait, one refusal must not stall a
        # run for minutes while a fan-out waits on it.
        retry_after_max: float = 60.0,
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
        self.max_tokens = max_tokens
        self.deadline = deadline
        self.retry_after_max = retry_after_max
        self._reasoning_refused = False
        self._sleep = time.sleep
        self._now = time.monotonic

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

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
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        return body

    def _send(self, body: dict) -> httpx.Response:
        """One request, read under a deadline for the whole reply.

        Read piece by piece so the clock can be checked between pieces: a
        provider that keeps sending a little at a time never trips a per-read
        timeout, however long the whole answer takes. A client that cannot
        stream is served the old way rather than refused -- the bound is worth
        having, but not at the price of the caller that passes its own client.
        """
        headers = self._headers()
        if self.deadline is None or not hasattr(self.client, "stream"):
            return self.client.post(self.provider.endpoint, headers=headers, json=body)
        limit = self._now() + self.deadline
        with self.client.stream(
            "POST", self.provider.endpoint, headers=headers, json=body
        ) as response:
            pieces = []
            for piece in response.iter_bytes():
                pieces.append(piece)
                if self._now() > limit:
                    raise httpx.ReadTimeout(
                        f"no complete reply within {self.deadline:g}s",
                        request=response.request,
                    )
            body = b"".join(pieces)
            # The pieces arrive decoded, so the headers describing how they
            # were packed no longer describe what is being handed on.
            headers = {
                name: value for name, value in response.headers.items()
                if name.lower() not in ("content-encoding", "content-length")
            }
            return httpx.Response(
                response.status_code,
                headers=headers,
                content=body,
                request=response.request,
            )

    def _retry_after(self, response: httpx.Response) -> float | None:
        """How long the provider asked us to wait, if it said so in seconds.

        The header also has a date form, which is not read: a clock skewed
        against the provider's turns it into an arbitrary wait, and the
        ordinary backoff is a safer answer than a wrong one.
        """
        try:
            seconds = float(response.headers.get("Retry-After", ""))
        except ValueError:
            return None
        return min(max(seconds, 0.0), self.retry_after_max)

    def _post(self, messages: Sequence[Message], model: str) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            delay = self.backoff * 2**attempt
            try:
                response = self._send(self._body(messages, model))
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
                wait = self._retry_after(response)
                # A refusal that names a time is one the provider expects to
                # pass -- credit held for requests already in flight, most of
                # all. One that names none is a verdict on this request, and
                # asking again only repeats it.
                retriable = response.status_code in RETRIABLE_STATUS or wait is not None
                if not retriable or last:
                    if response.is_error:
                        # The status alone says a request was refused, not
                        # why. The provider's own words are in the body, and
                        # without them a refusal that does not reproduce
                        # cannot be diagnosed afterwards at all.
                        raise httpx.HTTPStatusError(
                            f"{response.status_code} from {self.provider.name}: "
                            f"{response.text[:500]}",
                            request=response.request,
                            response=response,
                        )
                    return response
                delay = wait if wait is not None else delay
            self._sleep(delay)
        raise AssertionError("unreachable")

    def complete(self, messages: Sequence[Message], *, model: str) -> tuple[str, Spend]:
        response = self._post(messages, model)
        payload = response.json()
        raw = payload.get("usage") or {}
        usage = Spend(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
            cost=_cost(raw),
            cached_tokens=_detail(raw, "prompt_tokens_details", "cached_tokens"),
            reasoning_tokens=_detail(
                raw, "completion_tokens_details", "reasoning_tokens"
            ),
        )
        message = payload["choices"][0]["message"]
        text = (message.get("content") or "") + "".join(
            native_call(
                (call.get("function") or {}).get("name", "?"),
                (call.get("function") or {}).get("arguments", ""),
            )
            for call in message.get("tool_calls") or []
        )
        # Returned as a third item rather than folded into the text: it is not
        # part of the reply the loop acts on, and a sink that keeps records
        # wants it separable from the code the model actually wrote. Callers
        # written against the two-item shape keep working -- see `_reply` in
        # the engine.
        return text, usage, _reasoning_text(message)


class OpenRouterClient(ChatClient):
    provider = OPENROUTER


class DeepSeekClient(ChatClient):
    provider = DEEPSEEK


class AnthropicClient(ChatClient):
    """The Messages API, which is not shaped like the others.

    The system prompt travels beside the conversation rather than inside it,
    a reply ceiling is required rather than optional, the answer arrives as
    blocks rather than one string, and no price is ever reported — which the
    budget already reads as unknown rather than free.
    """

    provider = ANTHROPIC
    version = "2023-06-01"
    # Required by the API, so something has to be sent. Large enough for a
    # cell of code and the thinking around it; `max_tokens` overrides it.
    DEFAULT_MAX_TOKENS = 16000

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": self.version}

    def _body(self, messages: Sequence[Message], model: str) -> dict:
        spoken = [m for m in messages if m.get("role") != "system"]
        system = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "system"
        )
        body: dict = {
            "model": model,
            "messages": [dict(m) for m in spoken],
            "max_tokens": self.max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        if system:
            body["system"] = system
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def complete(self, messages: Sequence[Message], *, model: str):
        payload = self._post(messages, model).json()
        blocks = payload.get("content") or []
        text = "\n".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        ) + "".join(
            native_call(b.get("name", "?"), b.get("input", {}))
            for b in blocks if b.get("type") == "tool_use"
        )
        thinking = "\n".join(
            b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
        )
        raw = payload.get("usage") or {}
        cached = _number(raw.get("cache_read_input_tokens"))
        built = _number(raw.get("cache_creation_input_tokens"))
        # Cached and freshly cached tokens are input that was read but counted
        # apart, so a total that leaves them out understates the call.
        prompt = int(raw.get("input_tokens", 0) or 0) + int(cached or 0) + int(built or 0)
        completion = int(raw.get("output_tokens", 0) or 0)
        usage = Spend(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cost=None,
            cached_tokens=int(cached) if cached is not None else None,
            reasoning_tokens=None,
        )
        return text, usage, thinking or None


def make_client(name: str, **options) -> ChatClient:
    # A provider whose API differs in shape has a class of its own; the rest
    # share the OpenAI-shaped one and differ only in where they live.
    shaped = {"anthropic": AnthropicClient}
    if name in shaped:
        return shaped[name](**options)
    return ChatClient(provider=PROVIDERS[name], **options)


class ScriptedClient:
    def __init__(self, replies: Sequence[str]):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, model: str) -> tuple[str, Spend]:
        self.calls.append(list(messages))
        assert self.replies, "ScriptedClient ran out of scripted replies"
        return self.replies.pop(0), Spend()
