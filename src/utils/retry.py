# pyright: basic
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """
    Exponential backoff with optional jitter.

    delay(attempt) = min(max_delay_s, base_delay_s * (multiplier ** (attempt-1))) + jitter
    """

    base_delay_s: float = 0.5
    multiplier: float = 1.8
    max_delay_s: float = 20.0
    jitter_s: float = 0.2

    def delay_s(self, attempt: int) -> float:
        a = max(1, int(attempt))
        d = float(self.base_delay_s) * (float(self.multiplier) ** float(a - 1))
        d = min(float(self.max_delay_s), d)
        if self.jitter_s > 0:
            d += random.uniform(0.0, float(self.jitter_s))
        return max(0.0, float(d))


def async_retry(
    *,
    retries: int,
    backoff: BackoffPolicy | None = None,
    should_retry: Callable[[BaseException], bool] | None = None,
    log_prefix: str = "retry",
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Decorator for retrying async calls with exponential backoff.
    """

    backoff = backoff or BackoffPolicy()
    should_retry = should_retry or (lambda _e: True)
    total_attempts = int(retries) + 1

    def deco(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            last: BaseException | None = None
            for attempt in range(1, total_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except BaseException as e:
                    last = e
                    if attempt >= total_attempts or not should_retry(e):
                        raise
                    sleep_s = backoff.delay_s(attempt)
                    logger.warning(
                        "{} failed (attempt={}/{}). Sleeping {:.2f}s. err={}",
                        log_prefix,
                        attempt,
                        total_attempts,
                        sleep_s,
                        repr(e),
                    )
                    await asyncio.sleep(sleep_s)
            assert last is not None
            raise last

        return wrapped

    return deco

