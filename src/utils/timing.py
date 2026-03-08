from __future__ import annotations

import time


class FpsMeter:
    def __init__(self) -> None:
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        elapsed = max(now - self._last, 1e-6)
        self._last = now
        return 1.0 / elapsed
