"""Check causal optical credit and physically bounded joint-rate candidates."""

import numpy as np

from scripts.experimental.tt19_factory import RATE
from scripts.experimental.tt22_rally import ReturnMemory, cooperation_cost, rates_for, visual_return
from scripts.experimental.tt22_rally_trial import alternating


def test_optical_outcome_uses_the_first_visible_outgoing_bounce():
    y = (-.86, -.72, -.58, -.43, -.27, -.12, .03, .18, .33, .49, .62)
    z = (.20, .25, .28, .32, .34, .33, .30, .24, .16, .02, .064)
    samples = [(t, (.01, depth, height)) for t,(depth,height) in enumerate(zip(y, z), 14)]
    shot = dict(tick=10, prediction=dict(eta=4., landing=[0., .63], net=.36, hit=True, cost=0.))
    assert visual_return(samples[:-1], shot) is None
    observed = visual_return(samples, shot)
    assert observed['landing'] == [.01, .49] and observed['observed_tick'] == 24
    memory = ReturnMemory()
    assert memory.rank(shot['prediction']) == cooperation_cost(shot['prediction'])
    memory.observe(shot, observed)
    assert memory.bias()[1] < -.04 and memory.bias()[2] < 0.


def test_joint_commands_reach_targets_without_exceeding_public_ports():
    goals = np.full((2, 5, 5), .15)
    low, high, start = np.full(5, -.50), np.full(5, .10), np.zeros(5)
    tapes = rates_for(goals, start, low, high)
    assert np.max(abs(tapes)) <= 1.
    np.testing.assert_allclose(np.sum(tapes, axis=1) * RATE[:5], np.full((2, 5), .10))


def test_zero_correction_keeps_the_identical_edge_guard():
    shot = dict(landing=[.01, .79], net=.19, hit=True, cost=.4)
    control = ReturnMemory(guard_empty=True)
    learned = ReturnMemory([dict(residual=[0., 0., 0.])])
    assert control.rank(shot) == learned.rank(shot)
    assert control.rank(shot) > cooperation_cost(shot)


def test_rally_count_excludes_a_serve_and_restarts_after_same_side():
    sequence = [dict(kind='legal_serve', side=-1)]
    sequence += [dict(kind='legal_return', side=side) for side in (1, -1, -1, 1, -1)]
    assert alternating(sequence) == 3
