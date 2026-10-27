import random

from activity import MotionHistory, StateDebouncer, classify, describe
from zones import Zone

FPS = 15


def feed(history, seconds, start=(500, 500), velocity=(0, 0), jitter=0, scale=200, t0=0.0, seed=1):
    rng = random.Random(seed)
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        x = start[0] + velocity[0] * (t - t0) + rng.uniform(-jitter, jitter)
        y = start[1] + velocity[1] * (t - t0) + rng.uniform(-jitter, jitter)
        history.add(t, x, y, scale)


def test_speed_unknown_until_window_filled():
    h = MotionHistory(window_seconds=2.0)
    feed(h, 0.5)
    assert h.speed() is None


def test_jitter_alone_reads_as_still():
    # Stationary target with +/-4 px of box jitter per frame.
    h = MotionHistory(window_seconds=2.0)
    feed(h, 5, jitter=4)
    assert h.speed() < 0.05


def test_walking_reads_as_moving():
    # 200 px/s with a 200 px body = 1 body size per second.
    h = MotionHistory(window_seconds=2.0)
    feed(h, 5, velocity=(200, 0), jitter=4)
    assert 0.8 < h.speed() < 1.2


def test_speed_is_independent_of_distance_from_camera():
    near, far = MotionHistory(), MotionHistory()
    feed(near, 3, velocity=(300, 0), scale=300)
    feed(far, 3, velocity=(60, 0), scale=60)
    assert abs(near.speed() - far.speed()) < 0.05


def test_history_is_bounded():
    h = MotionHistory(window_seconds=2.0)
    feed(h, 60)
    assert len(h._obs) <= 2 * FPS + 2


def test_debouncer_ignores_short_flicker():
    d = StateDebouncer(hold_seconds=3)
    assert d.update("resting", 0) == "resting"
    assert d.update("active", 1) is None
    assert d.update("resting", 2) is None   # flicker over
    assert d.update("active", 3) is None    # new pending period starts at 3
    assert d.state == "resting"


def test_debouncer_accepts_sustained_change():
    d = StateDebouncer(hold_seconds=3)
    d.update("resting", 0)
    assert d.update("active", 1) is None   # pending since t=1
    assert d.update("active", 3) is None   # only 2 s so far
    assert d.update("active", 4) == "active"
    assert d.state == "active"


def test_classify_and_describe():
    food = Zone("food zone", "eating", [[0, 0], [10, 0], [10, 10], [0, 10]])
    assert classify(0.01, food, 0.15) == ("eating", "food zone")
    assert classify(0.9, food, 0.15) == ("active", "food zone")
    assert classify(0.01, None, 0.15) == ("resting", None)
    assert describe("person", ("eating", "food zone")) == "Person is eating in food zone"
    assert describe("person", ("active", None)) == "Person is active"
