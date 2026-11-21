import numpy as np

from scripts.experimental.tt19_flight import FlightLaw
from scripts.experimental.tt19_model_rollout import RolloutPlayer
from scripts.experimental.tt21_recovery import (ContextTimingMemory, NearSideRecovery, RallyContext, RecoveryRollout,
                                                 TimingMemory)


def test_timing_memory_restores_the_learned_condition_and_keeps_frozen_prior():
    memory = TimingMemory()
    for _ in range(3):
        memory.record(0, -1, 0.)
        memory.record(1, -1, .44)
        memory.record(2, -1, .08)

    assert TimingMemory(memory.state()).choose() == 1
    assert TimingMemory().choose() == 0


def test_optical_turnaround_requires_a_player_opening_and_an_incoming_ball():
    rally = RallyContext(-1)
    law = FlightLaw()
    blade = np.array((0., -1., .25))
    rally.observe(0, None, blade, False, law)
    rally.observe(23, (0., .5, .2), blade, False, law)
    assert not rally.outgoing
    rally.observe(30, (0., .26, .2), blade, True, law)
    rally.observe(31, (0., .17, .2), blade + [.040, 0., 0.], True, law)
    rally.observe(32, (0., .08, .2), blade + [.040, 0., 0.], True, law)
    assert rally.outgoing and rally.approach == 32
    assert abs(rally.blade_error - .04) < 1e-8


def test_contextual_updates_keep_supported_timing_decisions_separate():
    timing = ContextTimingMemory()
    for _ in range(3):
        timing.record(9, 0, -1, 0.)
        timing.record(9, 1, -1, .44)
        timing.record(12, 0, -1, .44)
        timing.record(12, 1, -1, 0.)
    assert timing.choose(9) == 1
    assert timing.choose(12) == 0
    assert timing.choose(None) == 0


def test_near_side_only_adds_one_window_and_preserves_factory_control(monkeypatch):
    monkeypatch.setattr(RolloutPlayer, 'command', lambda *_args, **_kwargs: 'factory')
    monkeypatch.setattr(RecoveryRollout, 'command', lambda *_args, **_kwargs: 'early')
    policy = NearSideRecovery.__new__(NearSideRecovery)
    policy.arm = 1
    assert policy.command(40, (0., .12, .3), -1) == 'factory'
    assert policy.command(41, (0., -.15, .25), -1) == 'early'
    assert policy.command(42, (0., -.24, .20), -1) == 'factory'
    policy.arm = 0
    assert policy.command(41, (0., -.15, .25), -1) == 'factory'
