"""Learn phase changes from matched RGB practice on disjoint rotary installations."""

from collections import deque

import numpy as np


PROFILES = (2, 1, 3)
PRACTICE_ARMS = ('fixed', 'fixed-early', 'fixed-late')
CUE_SCALE = np.array([.60, .45, .35, .45])


def practice_samples(report):
    """Pair the first planned contacts by start and score only their public evidence."""
    samples = []
    for station in report['held']:
        groups = station['conditions']
        arms = [groups[name] for name in PRACTICE_ARMS]
        if not all(arm.get('replayed_points') == len(arm['points']) for arm in arms):
            raise ValueError('matched timing practice must finish and replay all three arms')
        for rows in zip(*(arm['points'] for arm in arms), strict=True):
            if len({(row['seed'], row['phase']) for row in rows}) != 1:
                raise ValueError('matched timing practice schedules diverged')
            if not all(row['contact_timing'] for row in rows):
                continue
            attempts = [row['contact_timing'][0] for row in rows]
            if tuple(attempt['profile'] for attempt in attempts) != PROFILES:
                raise ValueError('matched timing practice did not use the expected phases')
            samples.append([attempts[0]['cues'], [a['score'] for a in attempts], [a['weight'] for a in attempts]])
    return samples


class PairedTimingMemory:
    """Use locally supported phase gains and adapt their scores to the current arm."""

    def __init__(self, practice=(), online=()):
        self.practice = list(practice)
        self.online = deque(online, maxlen=32)

    @staticmethod
    def _nearby(samples, cues):
        delta = (samples[:, 1:] - np.asarray(cues)[1:]) / CUE_SCALE
        return np.exp(-.5 * np.sum(delta**2, axis=1))

    def choose(self, cues, *, exploration=True):
        if not self.practice and exploration:
            observed = {row[0] for row in self.online}
            untried = next((phase for phase in PROFILES if phase not in observed), None)
            if untried is not None:
                return untried

        values = np.zeros(len(PROFILES))
        if self.practice:
            seen, scores, evidence = (np.asarray(items) for items in zip(*self.practice, strict=True))
            influence = self._nearby(seen, cues)[:, None] * np.minimum(evidence, evidence[:, :1])
            values = np.sum(influence * (scores - scores[:, :1]), axis=0) / (2. + influence.sum(axis=0))

        for index, profile in enumerate(PROFILES):
            recent = [r for r in self.online if r[0] == profile]
            if not recent:
                continue
            _, seen, residuals, evidence = zip(*recent, strict=True)
            influence = self._nearby(np.asarray(seen), cues) * evidence
            values[index] += influence @ residuals / (4. + influence.sum())
        values[0] += .045
        return PROFILES[int(np.argmax(values))]

    def observe(self, choice, cues, score, weight):
        index = PROFILES.index(choice)
        expected = .40
        if self.practice:
            seen, scores, evidence = (np.asarray(items) for items in zip(*self.practice, strict=True))
            influence = self._nearby(seen, cues) * evidence[:, index]
            expected = (.40 + influence @ scores[:, index]) / (1. + influence.sum())
        self.online.append([choice, np.asarray(cues).tolist(), float(score - expected), weight])

    def state(self):
        return dict(practice=self.practice, online=list(self.online))
