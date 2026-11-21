"""Retain and re-time useful strokes from the receiver's own visible outcomes."""

import numpy as np

from .tt19_contextual import optical_placement
from .tt19_model_rollout import observed_ball
from .tt19_shoot import COURT, HORIZON
from .tt21_continuation import phase_estimate


VARIANTS = (
    (0, 1., 0.),
    (1, 1.10, 0.),
    (-1, 1., 0.),
    (0, 1., .32),
    (0, 1., -.32),
    (-1, .89, 0.),
)
FEATURE_SCALE = np.array((.035, .050, .024, .045))
GUARD_VARIATION = np.array(((0.,) * 9,
                            (-.007, .009, -.008, -.025, -.06, .09, -34., 12., 0.),
                            (.007, -.009, .008, .025, .06, -.09, 34., -12., 0.)))


def strike_cue(law, estimate, tick):
    points, speeds = law.trajectory(*estimate, HORIZON, bounds=COURT)
    crossings = np.flatnonzero(points[1:, 1] <= -.89)
    if not len(crossings):
        return None
    index = int(crossings[0])
    fraction = np.clip((points[index, 1] + .89) / max(.001, points[index, 1] - points[index+1, 1]), 0., 1.)
    position = points[index] + fraction * (points[index+1] - points[index])
    velocity = speeds[index] + fraction * (speeds[index+1] - speeds[index])
    return dict(when=float(tick + index + fraction),
                features=[float(position[0]), float(position[2]), float(velocity[1]), float(velocity[2])])


def candidate(dry, estimate, cue, tick, variant):
    phase, amplitude, wrist = VARIANTS[variant]
    detail = dry.choose(phase_estimate(dry.law, estimate, phase))
    if detail is None or (not detail['legal'] and min(detail['robustness']) > 2.8 and estimate[0][1] > -.51):
        return None
    steps = np.asarray(dry.tapes[detail['index'], :detail['duration']], dtype=float).copy() * amplitude
    finish = max(1, int(np.ceil(detail['eta'])) - 1)
    lo = max(0, finish - 3)
    steps[lo:finish, 3] += wrist / (finish - lo)
    return dict(cue=cue, start=tick, rates=np.clip(steps, -1., 1.).tolist(),
                factory_index=detail['index'], factory_eta=detail['eta'], variant=variant)


def visual_credit(sightings, reward, stroke, law):
    if stroke is None:
        return .2 * reward
    arc = [(tick, point) for tick, point in sightings if stroke['start']-2 <= tick <= stroke['start']+43
           and abs(point[0]) < .57 and -.035 < point[2] < .75]
    return optical_placement(arc, -1, reward, law, None)


def guard_prefers_recall(legal_counts, mean_costs):
    stock, recall = legal_counts
    return bool(recall >= 2 and recall > stock and mean_costs[1] + .25 < mean_costs[0])


def compare_proposals(dry, estimate, tick, stock, recalled):
    """Give both actual command schedules the same independent model of the ball."""
    tapes = np.zeros((2, HORIZON, 5))
    duration = []
    for i, proposal in enumerate((stock, recalled)):
        wait = proposal['start'] - tick
        end = min(HORIZON, wait + len(proposal['rates']))
        tapes[i, wait:end] = proposal['rates'][:end - wait]
        duration.append(end)
    count = len(GUARD_VARIATION)
    ball = observed_ball(dry.history, estimate, dry.law)
    results = dry.sim.predict(np.repeat(tapes, count, axis=0), np.repeat(duration, count),
                              ball, np.tile(GUARD_VARIATION, (2, 1)))
    legal = np.asarray([result['legal'] for result in results]).reshape((2, count))
    costs = np.asarray([result['cost'] for result in results]).reshape((2, count))
    legal_counts, mean_costs = legal.sum(axis=1), costs.mean(axis=1)
    keep = guard_prefers_recall(legal_counts, mean_costs)
    return recalled if keep else stock, dict(tick=tick, kept_recall=keep, predicted_legal=legal.tolist(),
                                             predicted_cost=costs.tolist())


class StrikeMemory:
    """Keep motor tapes with observed flight cues, without a live-state input."""

    def __init__(self, rows=()):
        self.rows = list(rows)

    def observe(self, stroke, value, token):
        if stroke is not None and value >= 1.05:
            self.rows.append(dict(stroke=stroke, value=float(value), token=token))

    def retrieve(self, cue, tick):
        nearby = []
        for row in self.rows:
            stroke = row['stroke']
            offset = cue['when'] - stroke['cue']['when']
            wanted = stroke['start'] + int(round(offset))
            if not tick - 1 <= wanted <= tick + 5:
                continue
            distance = np.linalg.norm((np.asarray(cue['features']) - stroke['cue']['features']) / FEATURE_SCALE)
            if distance <= 3.5:
                quality = row['value'] - .34 * distance - .12 * max(0, tick - wanted)
                nearby.append((quality, row, wanted))
        if not nearby:
            return None
        _, row, wanted = max(nearby, key=lambda item: item[0])
        return dict(row['stroke'], cue=cue, start=max(tick, wanted), recalled=row['token'])
