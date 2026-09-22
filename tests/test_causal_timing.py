import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from causal_timing import sensory_history, causal_window


def test_future_event_cannot_change_an_issued_gate():
    event = np.array([-1., 0., 1., 0., -1., 0., 1.])
    changed = event.copy(); changed[6] = 0
    state = sensory_history(event, .8, steps=1)
    other = sensory_history(changed, .8, steps=1)
    a, reference = causal_window(state, event, .8)
    b, _ = causal_window(other, changed, .8)
    np.testing.assert_array_equal(a[:6], b[:6])
    assert reference[2] == abs(state[1])
    assert reference[1] == 0


def test_constant_interval_is_acquired_without_labels_or_resets():
    event = np.zeros(100)
    event[10::3] = np.resize([-1., 1.], len(event[10::3]))
    state = sensory_history(event, .7, steps=1)
    gate, _ = causal_window(state, event, .7)
    np.testing.assert_array_equal(gate[30:-1], event[31:] != 0)


def test_sensory_history_applies_impulse_before_recording():
    np.testing.assert_allclose(sensory_history(np.array([1., 0., -1.]), .5, steps=2),
                               [30., 7.5, -28.125])


def test_omission_has_identical_prefix_but_changes_later_reference():
    events = np.zeros(80)
    events[10::3] = np.resize([-1., 1.], len(events[10::3]))
    omitted = events.copy(); omitted[[40, 43]] = 0
    a, ar = causal_window(sensory_history(events, .7, 1), events, .7)
    b, br = causal_window(sensory_history(omitted, .7, 1), omitted, .7)
    np.testing.assert_array_equal(a[:40], b[:40])
    np.testing.assert_array_equal(ar[:40], br[:40])
    assert a[39] and b[39]
    assert br[46] != ar[46]
