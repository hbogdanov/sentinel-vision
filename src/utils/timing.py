from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


class FpsMeter:
    def __init__(self) -> None:
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        elapsed = max(now - self._last, 1e-6)
        self._last = now
        return 1.0 / elapsed


@dataclass(slots=True)
class RollingTimingStats:
    window_size: int = 120
    _values: dict[str, deque[float]] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            bucket = self._values.setdefault(stage, deque(maxlen=max(1, self.window_size)))
            bucket.append(elapsed)

    def summary_ms(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        for stage, values in self._values.items():
            if values:
                summary[stage] = round((sum(values) / len(values)) * 1000.0, 3)
        return summary

    def stage_seconds(self, stage: str) -> float:
        values = self._values.get(stage)
        if not values:
            return 0.0
        return sum(values) / len(values)
