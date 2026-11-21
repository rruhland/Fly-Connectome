"""Estimate fingertip release time from the painted two-button panel."""

from dataclasses import dataclass
from statistics import median

_MIN_RELEASES = 2
_RECENT_RELEASES = 12
_MAX_EXTRA_STOP_SECONDS = .25


@dataclass(frozen=True)
class VisualRelease:
    button: int
    request_tick: int
    seen_tick: int

    @property
    def ticks(self):
        return self.seen_tick - self.request_tick


class ReleaseTiming:
    """Record delay from steering away until the fingertip leaves the button.

    The common controller supplies its estimate of which button the painted
    fingertip occupies. Steering back to that button before visual release
    cancels the sample. The fixed control records the same kind of evidence.
    """

    def __init__(self, tick_seconds, *, learn):
        self.tick_seconds = tick_seconds
        self.learn = learn
        self.held = None
        self._pending = None
        self.releases = []

    @property
    def stop_seconds(self):
        if self.held is None or not self.learn or len(self.releases) < _MIN_RELEASES:
            return 0.
        recent = self.releases[-_RECENT_RELEASES:]
        return min(_MAX_EXTRA_STOP_SECONDS,
                   self.tick_seconds * median(event.ticks for event in recent))

    def observe(self, tick, held):
        if self._pending is not None and held != self.held:
            self.releases.append(VisualRelease(self.held, self._pending, tick))
            self._pending = None
        self.held = held

    def request(self, tick, target):
        if self.held is None or target == self.held:
            self._pending = None
        elif self._pending is None:
            self._pending = tick
