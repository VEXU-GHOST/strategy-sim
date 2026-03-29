"""Lightweight timing accumulator with context-manager sections."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class PerfAccum:
    """Accumulates ``perf_counter`` deltas per named section.

    Usage::

        perf = PerfAccum()
        for frame in frames:
            with perf.section("render"):
                img = render(frame)
            with perf.section("encode"):
                enc.feed(img)
        print(perf.report_ms(len(frames)))
    """

    __slots__ = ("_totals",)

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self._totals[name] = self._totals.get(name, 0.0) + (time.perf_counter() - t)

    def add(self, name: str, seconds: float) -> None:
        """Manually accumulate *seconds* to *name*."""
        self._totals[name] = self._totals.get(name, 0.0) + seconds

    def seconds(self, name: str) -> float:
        return self._totals.get(name, 0.0)

    def total(self) -> float:
        return sum(self._totals.values())

    def report_ms(
        self,
        n: int,
        precision: int = 2,
        exclude: frozenset[str] = frozenset(),
    ) -> str:
        """Format as ``'key1=X.XX key2=Y.YY'`` in ms per *n*."""
        parts: list[str] = []
        for k, v in self._totals.items():
            if k not in exclude:
                parts.append(f"{k}={v / n * 1000:.{precision}f}")
        return " ".join(parts)

    def reset(self) -> None:
        self._totals.clear()
