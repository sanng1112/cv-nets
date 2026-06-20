"""Timing / benchmarking utilities."""

import time
from typing import Optional


def format_duration(seconds: float) -> str:
    """Format a duration (seconds) into a human-readable string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{seconds:.3f}s"


class Timer:
    """Context manager for measuring elapsed time.

    Usage
    -----
    >>> with Timer() as t:
    ...     do_something()
    >>> print(t.elapsed)
    """

    def __init__(self, verbose: bool = False):
        self._start: Optional[float] = None
        self._elapsed: float = 0.0
        self.verbose = verbose

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start
        if self.verbose:
            from cvnets.utils.logger import info
            info(f"Elapsed: {format_duration(self._elapsed)}")

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def reset(self) -> None:
        self._start = None
        self._elapsed = 0.0
