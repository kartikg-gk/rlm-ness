"""Limits shared across a recursion tree."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .providers import Spend


class AllowanceSpent(Exception):
    pass


class TimeUp(AllowanceSpent):
    pass


class Abandoned(Exception):
    """Raised when work is abandoned, not when a limit is reached."""


@dataclass
class Allowance:
    max_depth: int = 3
    max_calls: int = 60
    max_cost: float = 0.5
    max_live: int = 16
    max_seconds: float | None = None
    # Token ceilings are off unless asked for. Tokens are what a provider
    # always reports, so these hold where a price is never sent.
    max_completion_tokens: int | None = None
    max_prompt_tokens: int | None = None
    calls: int = 0
    cost: float = 0.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    unpriced_calls: int = 0
    # A batch reserves and settles from several threads at once.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    _cancelled: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    _slots: threading.Semaphore | None = field(default=None, repr=False, compare=False)
    _started: float = field(default_factory=time.monotonic, repr=False, compare=False)

    def __post_init__(self):
        if self._slots is None:
            self._slots = threading.Semaphore(max(1, self.max_live))

    @classmethod
    def from_config(cls, config) -> "Allowance":
        return cls(
            max_depth=config.max_depth,
            max_calls=config.max_calls,
            max_cost=config.max_cost,
            max_live=getattr(config, "max_live", 16),
            max_seconds=getattr(config, "max_seconds", None),
            max_completion_tokens=getattr(config, "max_completion_tokens", None),
            max_prompt_tokens=getattr(config, "max_prompt_tokens", None),
        )

    def reserve(self) -> None:
        """Claim one call, or refuse. Checking and claiming happen together.

        Doing them separately is what let a whole batch pass the check before
        any of it had spent, and then overrun the limit by the width of the
        batch.
        """
        if self._cancelled.is_set():
            raise Abandoned("work was abandoned")
        with self._lock:
            if self.max_seconds is not None:
                elapsed = time.monotonic() - self._started
                if elapsed >= self.max_seconds:
                    raise TimeUp(
                        f"{elapsed:.1f}s reaches the limit of {self.max_seconds}s"
                    )
            if self.calls >= self.max_calls:
                raise AllowanceSpent(
                    f"{self.calls} calls reaches the limit of {self.max_calls}"
                )
            if self.cost >= self.max_cost:
                raise AllowanceSpent(
                    f"cost {self.cost:.4f} reaches the limit of {self.max_cost}"
                )
            if (
                self.max_completion_tokens is not None
                and self.completion_tokens >= self.max_completion_tokens
            ):
                raise AllowanceSpent(
                    f"{self.completion_tokens} completion tokens reaches the "
                    f"limit of {self.max_completion_tokens}"
                )
            if (
                self.max_prompt_tokens is not None
                and self.prompt_tokens >= self.max_prompt_tokens
            ):
                raise AllowanceSpent(
                    f"{self.prompt_tokens} prompt tokens reaches the limit of "
                    f"{self.max_prompt_tokens}"
                )
            self.calls += 1

    def settle(self, usage: Spend) -> None:
        """Record what a reserved call actually cost.

        A call's price is only known once it has been made, so cost can still
        pass its limit by whatever the calls already in flight turn out to
        cost. It cannot be exceeded by anything started afterwards.

        A provider that reports no price leaves `max_cost` unable to fire at
        all. Those calls are counted separately rather than treated as free, so
        the gap can be reported instead of being mistaken for a small bill.
        """
        with self._lock:
            self.completion_tokens += usage.completion_tokens
            self.prompt_tokens += usage.prompt_tokens
            if usage.cost is None:
                self.unpriced_calls += 1
            else:
                self.cost += usage.cost

    @property
    def cost_is_complete(self) -> bool:
        return self.unpriced_calls == 0

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def claim_slots(self, wanted: int) -> int:
        """Take up to `wanted` slots without ever blocking. May return zero.

        Never blocking is the point: a caller that gets nothing runs the work
        itself instead of waiting, so a tree already holding every slot at one
        depth still makes progress rather than deadlocking against its own
        children. Release exactly what was returned.
        """
        taken = 0
        while taken < wanted and self._slots.acquire(blocking=False):
            taken += 1
        return taken

    def release_slots(self, count: int) -> None:
        for _ in range(count):
            self._slots.release()

    def can_recurse(self, depth: int) -> bool:
        return depth + 1 <= self.max_depth
