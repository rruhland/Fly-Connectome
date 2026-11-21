"""Preserve a camera-planned stroke across a brief blind interval near contact."""

from dataclasses import dataclass

import numpy as np

from .tt6_rotary_adapt import RotaryPlayer, STROKES, Stroke, StrokeMemory


@dataclass(frozen=True)
class FollowThrough:
    frames: float = 5.
    extension: float = .07
    lift: float = 0.
    rise: float = 0.


class ContactPlayer(RotaryPlayer):
    """Continue a bounded stroke from the last visually supported intercept."""

    def __init__(self, *args, follow=FollowThrough(), **kwargs):
        self.follow = follow
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.last_stroke = None
        self.held_frames = 0

    def _stroke_goals(self, estimate):
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        tick = self.scene.tick
        if horizon is not None:
            delay = self.servo.observer.delay if self.use_delay else 0.
            if self.follow.lift or self.follow.rise:
                offset = self.follow.lift - np.clip(self.follow.rise * (horizon - delay), -.06, .04)
                next_offset = self.follow.lift - np.clip(self.follow.rise * (horizon - delay - 1.), -.06, .04)
                goal = goal + (0., 0., offset)
                lead = lead + (0., 0., next_offset)
            self.last_stroke = (tick, tick + horizon, lead.copy(), pitch) if horizon < 5. else None
            return goal, lead, pitch, horizon
        if self.last_stroke is None or self.follow.frames <= 0.:
            return goal, lead, pitch, horizon
        last_tick, expected, saved_lead, saved_pitch = self.last_stroke
        if tick >= expected + self.follow.frames or tick - last_tick > self.follow.frames:
            self.last_stroke = None
            return goal, lead, pitch, horizon
        self.held_frames += 1
        target = saved_lead.copy()
        target[1] += min(self.follow.extension, self.stroke_speed * .55 * (tick - last_tick))
        return target, target.copy(), saved_pitch, expected - tick

    def finish(self, reward):
        result = super().finish(reward)
        result['follow_through_frames'] = self.held_frames
        return result


HIGH_LOFT = (Stroke(.073, .19, .95, .04, .55), FollowThrough(5., .07, .07, .022))
BASELINE_CONTACT = (STROKES[1], FollowThrough(0.))


class AdaptiveContactPlayer(ContactPlayer):
    """Choose between two strokes from recent optical landings and point rewards."""

    profiles = (BASELINE_CONTACT, HIGH_LOFT)

    def __init__(self, *args, **kwargs):
        self.profile_memory = StrokeMemory()
        self.profile_memory.counts = np.zeros(len(self.profiles), dtype=int)
        self.profile_memory.totals = np.zeros(len(self.profiles))
        super().__init__(*args, **kwargs)

    def reset(self):
        self.profile_index = self.profile_memory.choose(explore=True)
        self.stroke_settings, self.follow = self.profiles[self.profile_index]
        super().reset()

    def finish(self, reward):
        result = super().finish(reward)
        score = self.profile_memory.observe(self.profile_index, result['stroke_observation'], reward)
        result.update(profile_index=self.profile_index, profile_score=score, profile_memory=self.profile_memory.state())
        return result
