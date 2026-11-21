import numpy as np

from scripts.experimental.tt19_fast_play import CachedRollout, TimedRolloutPlayer


class Simulator:
    def __init__(self):
        self.batches = []

    def predict(self, tapes, durations, observation, offsets):
        self.batches.append(len(tapes))
        return [float(tape.sum() + duration + offset.sum() + observation)
                for tape, duration, offset in zip(tapes, durations, offsets, strict=True)]


def test_rollout_cache_reuses_exact_inputs_only_within_one_decision():
    simulator = Simulator()
    cache = CachedRollout(simulator, enabled=True)
    tapes = np.arange(30.).reshape((3, 2, 5))
    durations = np.array([2, 3, 4])
    offsets = np.zeros((3, 9))
    initial = cache.predict(tapes[:2], durations[:2], 5., offsets[:2])
    assert cache.predict(tapes, durations, 5., offsets)[:2] == initial
    offsets[0, 0] = .01
    assert cache.predict(tapes, durations, 5., offsets)[0] != initial[0]
    assert simulator.batches == [2, 1, 1]
    assert cache.simulated == 4
    assert cache.reused == 4
    cache.reset()
    assert cache.predict(tapes[:1], durations[:1], 10., np.zeros((1, 9)))[0] == initial[0] + 5.


def test_small_search_requires_nominal_legality_and_both_optical_scenarios():
    decision = dict(legal=True, robustness=[.4, .5, .3])
    assert TimedRolloutPlayer._supported(decision)
    assert not TimedRolloutPlayer._supported(dict(decision, legal=False))
    assert not TimedRolloutPlayer._supported(dict(decision, robustness=[.4, .5, 3.]))
    assert not TimedRolloutPlayer._supported(None)
