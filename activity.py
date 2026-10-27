"""Activity classification for tracked detections.

MotionHistory measures movement over a sliding time window by comparing the
average position at the start of the window with the average at the end.
Speed is expressed in body sizes per second (displacement / sqrt(box area)),
so the same threshold applies at any distance from the camera.

StateDebouncer accepts a new state only after it has persisted for a minimum
duration.
"""
import math
from collections import deque
from dataclasses import dataclass, field


class MotionHistory:
    """Recent (time, x, y, scale) observations for one tracked animal."""

    def __init__(self, window_seconds=2.0):
        self.window = window_seconds
        self._obs = deque()

    def add(self, t, x, y, scale):
        self._obs.append((t, x, y, max(scale, 1.0)))
        # Drop observations older than the window.
        while len(self._obs) > 2 and t - self._obs[1][0] >= self.window:
            self._obs.popleft()

    @property
    def last_seen(self):
        return self._obs[-1][0] if self._obs else None

    def speed(self):
        """Return speed in body sizes per second over the window, or None if
        less than half the window has been observed."""
        if len(self._obs) < 2:
            return None
        span = self._obs[-1][0] - self._obs[0][0]
        if span < self.window * 0.5:
            return None

        # Average the first and last quarter of the observations.
        k = max(1, len(self._obs) // 4)
        head = list(self._obs)[:k]
        tail = list(self._obs)[-k:]
        t0, x0, y0 = (sum(o[i] for o in head) / k for i in range(3))
        t1, x1, y1 = (sum(o[i] for o in tail) / k for i in range(3))
        dt = t1 - t0
        if dt <= 0:
            return None
        scales = sorted(o[3] for o in self._obs)
        scale = scales[len(scales) // 2]  # median box size
        return math.dist((x0, y0), (x1, y1)) / scale / dt


class StateDebouncer:
    """Commits a new state once it has been observed continuously for
    `hold_seconds`. The first state is committed immediately."""

    def __init__(self, hold_seconds=3.0):
        self.hold = hold_seconds
        self.state = None
        self._pending = None
        self._pending_since = None

    def update(self, candidate, now):
        """Process a candidate state. Return the new state if it was
        committed, otherwise None."""
        if self.state is None:
            self.state = candidate
            return candidate
        if candidate == self.state:
            self._pending = None
            return None
        if candidate != self._pending:
            self._pending, self._pending_since = candidate, now
        if now - self._pending_since >= self.hold:
            self.state, self._pending = candidate, None
            return candidate
        return None


def classify(speed, zone, still_threshold):
    """Return (activity, zone_name) for a speed and the zone it occurred in."""
    still = speed < still_threshold
    if zone is None:
        return ("resting" if still else "active"), None
    return (zone.still_label if still else "active"), zone.name


def describe(label, state):
    activity, zone_name = state
    text = f"{label.capitalize()} is {activity}"
    return f"{text} in {zone_name}" if zone_name else text


@dataclass
class TrackState:
    history: MotionHistory
    debouncer: StateDebouncer
    last_logged: float = field(default=-math.inf)
