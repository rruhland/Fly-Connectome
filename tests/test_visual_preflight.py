"""Frozen feasibility must preserve arrival delays and reject tonic-only evidence."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from visual_preflight import delayed_traces, supported_response


def test_signed_trace_arrives_after_fixed_delay_and_decays():
    spikes = np.array([[True, False], [False, True], [False, False], [False, False]])
    trace = delayed_traces(spikes, np.array([0, 1]), np.array([2, 1]),
                           np.array([-1, 1]), np.array([.5, .25]))
    np.testing.assert_array_equal(trace, [[0, 0], [0, 0], [-1, 1], [-.5, .25]])


def test_tonic_trace_wrong_sign_or_cancellation_cannot_pass_preflight():
    assert not supported_response(-1, -.2, -.2, np.array([-.5]), np.array([-.5]))
    assert not supported_response(-1, .2, .1, np.array([.5]), np.array([0.]))
    assert not supported_response(-1, .1, .2, np.array([-.5]), np.array([0.]))
    assert supported_response(-1, -.2, -.1, np.array([-.5]), np.array([0.]))
    assert supported_response(1, .2, .1, np.array([.5]), np.array([0.]))
