"""Try a cheap robust shortlist before repeating the full CAD search."""

from time import perf_counter_ns, process_time_ns

import numpy as np

from .tt19_model_rollout import RolloutPlayer


SMALL = 8
FULL = 32
ROBUST = 2.8


class CachedRollout:
    """Retain exact simulation results only for the current optical decision."""

    def __init__(self, simulator, *, enabled):
        self.simulator = simulator
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.cache = {}
        self.simulated = self.reused = self.batches = 0

    def predict(self, tapes, durations, observation, offsets):
        if not self.enabled:
            self.simulated += len(tapes)
            self.batches += 1
            return self.simulator.predict(tapes, durations, observation, offsets)
        keys = [(tape.tobytes(), int(duration), offset.tobytes())
                for tape, duration, offset in zip(tapes, durations, offsets, strict=True)]
        missing = {}
        for index, key in enumerate(keys):
            if key not in self.cache:
                missing.setdefault(key, index)
        if missing:
            rows = list(missing.values())
            output = self.simulator.predict(tapes[rows], durations[rows], observation, offsets[rows])
            self.cache.update(zip(missing, output, strict=True))
            self.simulated += len(rows)
            self.batches += 1
        self.reused += len(keys) - len(missing)
        return [self.cache[key] for key in keys]

    def close(self):
        self.simulator.close()


class TimedRolloutPlayer(RolloutPlayer):
    """Measure each optical planning call and expand uncertain shortlists."""

    def __init__(self, catalog, law, response, *, adaptive):
        super().__init__(catalog, law, response, count=FULL)
        self.sim = CachedRollout(self.sim, enabled=adaptive)
        self.adaptive = adaptive
        self.calls = []

    def choose(self, estimate):
        self.sim.reset()
        wall, cpu = perf_counter_ns(), process_time_ns()
        fallback = False
        try:
            self.count = SMALL if self.adaptive else FULL
            decision = super().choose(estimate)
            if self.adaptive and self.sim.simulated and not self._supported(decision):
                fallback = True
                self.count = FULL
                decision = super().choose(estimate)
        finally:
            self.count = FULL
        elapsed_cpu, elapsed_wall = process_time_ns() - cpu, perf_counter_ns() - wall
        self.calls.append(dict(tick=self.history[-1][0], fallback=fallback, wall_ns=elapsed_wall,
                               cpu_ns=elapsed_cpu, full_trials=self.sim.simulated,
                               cached_trials=self.sim.reused, full_batches=self.sim.batches,
                               optical=[[int(tick), None if xyz is None else np.asarray(xyz).tolist()]
                                        for tick, xyz in self.history[-8:]],
                               estimate=[np.asarray(value).tolist() for value in estimate], decision=decision))
        return decision

    @staticmethod
    def _supported(decision):
        return (decision is not None and decision['legal']
                and max(decision['robustness']) <= ROBUST)
