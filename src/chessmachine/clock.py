"""A one-sided match clock.

It charges the *human* only — for the time they spend thinking before their
prompt is ready. The machine's own work (speech recognition, the SLM, crane
actuation, the spoken explanation) happens off this clock by design: that
processing time is exactly what gives the player a moment to plan their reply,
so it must not count against them. See docs / the move-commentary design notes.
"""
from __future__ import annotations

import time
from collections.abc import Callable


class MatchClock:
    def __init__(self, now: Callable[[], float] = time.monotonic):
        self._now = now
        self.user_seconds = 0.0
        self._started_at: float | None = None

    def start_user(self) -> None:
        """The human's turn to think begins."""
        if self._started_at is None:
            self._started_at = self._now()

    def stop_user(self) -> float:
        """Their prompt is ready; bank the elapsed think time. Returns it."""
        if self._started_at is None:
            return 0.0
        elapsed = self._now() - self._started_at
        self.user_seconds += elapsed
        self._started_at = None
        return elapsed

    def reset(self) -> None:
        self.user_seconds = 0.0
        self._started_at = None

    def spoken(self) -> str:
        """The accumulated user time as a speakable phrase."""
        return self.format_seconds(self.user_seconds)

    @staticmethod
    def format_seconds(seconds: float) -> str:
        total = int(seconds)
        minutes, secs = divmod(total, 60)
        if minutes:
            mtxt = f"{minutes} minute{'s' if minutes != 1 else ''}"
            return f"{mtxt} {secs} second{'s' if secs != 1 else ''}" if secs else mtxt
        return f"{secs} second{'s' if secs != 1 else ''}"
