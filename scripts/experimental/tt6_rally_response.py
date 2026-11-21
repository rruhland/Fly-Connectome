"""Use the first visible rebound of a rally to adjust a subsequent stroke."""
from copy import deepcopy

from .tt6_continuous import ContinualPlayer


class RallyResponsePlayer(ContinualPlayer):
    def reset(self):
        super().reset()
        self.adjusted_at = None
        self.optical_candidates = 0
        self.approach_tick = None

    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        frame = self.trace[-1]
        if self.adjusted_at is not None:
            return command
        tick, horizon = frame['tick'], frame['horizon']
        if horizon is not None and 0 <= horizon <= 2.5 and self.approach_tick is None:
            self.approach_tick = tick
        if self.approach_tick is None or tick - self.approach_tick not in (9, 14, 21):
            return command
        # Keep this inference local to the rally. The ordinary between-point
        # rule receives the actual public score once the entire point is over.
        local = deepcopy(self.response)
        local.reacquisition = True
        start = max(0, len(self.trace) - (tick - self.approach_tick + 14))
        result = local.finish([item['ball'] for item in self.trace[start:]], reward=0.)
        self.optical_candidates += 1
        if result['visual_impact'] is not None and result['response'] is not None:
            self.adjusted_at = tick
            self.stroke_speed = local.speed
        return command

    def finish(self, reward):
        details = super().finish(reward)
        details['within_rally'] = dict(checked=self.optical_candidates, adjusted_at=self.adjusted_at,
                                       speed=self.stroke_speed)
        return details
