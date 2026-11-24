import numpy as np

from scripts.experimental.tt29_vision import Flight
from scripts.experimental.tt41_anticipate import GateEye, physical_projection, poly_arc
from scripts.experimental.tt41_anticipate_fit import average_precision, lock, operation


def fit(t, y, vy, *, bounced=False):
    return Flight(np.array([0., y, 1.]), np.array([0., vy, 0.]), t, .5, .2, .01, 5, bounced)


def test_short_far_turn_restarts_latches_without_a_native_hit():
    eye = GateEye()
    ok = lambda t, y, v: [m['gate'] for m in eye.step(fit(t, y, v), t, None)]
    assert ok(.1, -.6, 3.) == ['out_-0.70']
    assert ok(.3, .3, 3.) == ['out_-0.10']
    assert not ok(.36, .45, 3.) == []
    # First accepted fit is already on the near side of the net.
    assert 'first_in' in ok(.60, -.18, -4.)
    assert 'first_in' not in ok(.70, -.55, -4.)
    ok(.85, -1.22, -4.)
    assert 'out_-0.70' in ok(1.13, -.64, 3.5)


def test_future_height_is_fit_from_the_optical_rebound():
    dt = 1 / 30
    raw = []
    for i in range(-5, 8):
        t = i * dt
        z = .784 + (2.35 if t >= 0 else -3.) * t - 4.905 * t * t
        raw.append([1+t, .08+.12*t, -.75-3.5*t, z])
    a = poly_arc(raw, 1.25, dt=dt)
    assert a is not None
    assert .99 < a['height'] < 1.12
    assert abs(a['optical_bounce_s']-1.) <= dt
    assert a['n'] >= 4


def test_missing_or_late_camera_gate_counts_against_recall():
    z = np.array([.96, .94, .98, 1.03, 1.04, 1.05])
    scores = np.array([.01, .012, np.nan, .02, .03, .04])
    low = z < .99
    assert np.isclose(average_precision(scores, low), 2/3)
    assert lock(scores, low, .60) is None
    s = operation(scores, low, .024, np.array([.95, .50, np.nan, 1.2, 1.2, 1.2]))
    assert s['flagged'] == 3 and s['true_low'] == 2 and s['eligible_recall'] == 2/3
    assert s['deadline']['0.88']['recall_in_time'] == 1/3


def test_unphysical_elapsed_projection_abstains():
    assert physical_projection([-.12, 1.42, 1., .10, -1.16, -.13], False) is None
