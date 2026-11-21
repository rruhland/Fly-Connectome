"""Credit a small physical stroke-timing correction only after the opening shot."""

import numpy as np

from .tt19_model_rollout import RolloutPlayer
from .tt19_shoot import COURT


PHASES = (0, 1, -1)
FALLBACKS = {'retry with earlier estimate': -1, 'retry with later estimate': 1}


def phase_estimate(law, estimate, frames):
    """Rephase a local RGB flight estimate by one control frame."""
    position, velocity = map(np.asarray, estimate)
    if frames == 1:
        points, speeds = law.trajectory(position, velocity, 1, bounds=COURT)
        return points[-1], speeds[-1]
    if frames == -1:
        acceleration = -law.drag * np.linalg.norm(velocity) * velocity
        acceleration[2] -= law.gravity
        earlier = position - velocity + .5 * acceleration
        if earlier[2] >= .020:
            return earlier, velocity - acceleration
    return estimate


class ContinuationPlayer(RolloutPlayer):
    """Keep opening and recovery commands and calibrate timing on later strokes."""

    phase = 0

    def reset(self):
        super().reset()
        self.followup = False
        self.followup_planning = []

    def choose(self, estimate):
        if not self.followup:
            return super().choose(estimate)
        if self.phase in FALLBACKS:
            nominal = super().choose(estimate)
            phase = FALLBACKS[self.phase]
            if nominal is not None and nominal['legal']:
                phase, result = 0, nominal
            else:
                alternate = super().choose(phase_estimate(self.law, estimate, phase))
                if alternate is None:
                    phase, result = 0, nominal
                else:
                    result = alternate
        else:
            phase = self.phase
            result = super().choose(phase_estimate(self.law, estimate, phase))
        tick = self.history[-1][0]
        self.followup_planning.append(dict(tick=tick, selected=None if result is None else result['index']))
        if result is not None and phase != 0:
            result['continuation_phase'] = phase
        return result


def learn_phase(rows):
    """Rank balanced treatments by actual legal return, contact, then point credit."""
    totals = {}
    for phase in PHASES:
        trials = [row for row in rows if row['phase'] == phase]
        totals[phase] = tuple(sum(bool(row['credit'][kind]) for row in trials)
                              for kind in ('legal', 'contact', 'win'))
    # Stable max keeps the existing stroke on exact ties.
    return max(PHASES, key=totals.__getitem__), {str(phase): credit for phase, credit in totals.items()}


def learn_fallback(rows):
    """Use contact credit from practice replies without a nominal legal plan."""
    uncertain = {row['seed'] for row in rows if row['phase'] == 0 and row['first_rival_reply'] is not None
                 and not any(plan['legal'] for plan in row['plans'] if plan['tick'] >= row['first_rival_reply'])}
    selected, scores = learn_phase([row for row in rows if row['seed'] in uncertain])
    options = {frames: choice for choice, frames in FALLBACKS.items()}
    return options.get(selected, 0), dict(scores=scores, starts=sorted(uncertain), selected_phase=selected)
