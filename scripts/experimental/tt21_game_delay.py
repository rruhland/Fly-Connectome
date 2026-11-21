"""Estimate drive latency from the RGB consequences of ordinary game actions."""

from time import perf_counter

import numpy as np

from .tt4_arena import mujoco
from .tt19_match_player import MatchPlayer
from .tt19_model_rollout import DrySimulator
from .tt19_view_trial import CAMERAS, WRIST_CAMERA
from .tt20_fixture import TT20Arena
from .tt20_latency import LatencyPlayer, identify_delay
from .tt21_body_state import CommandTwin


MAX_DELAY = 6


def motion_sample(eye, rgb):
    """Use red blade silhouette changes or off-blade wrist scene changes."""
    height, width = rgb.shape[:2]
    if eye == 'left':
        red, green, blue = np.moveaxis(rgb[::2, ::2].astype(np.int16), -1, 0)
        chroma = red - np.minimum(green, blue)
        return ((chroma >= 55) & (red - np.maximum(green, blue) > 40)
                & (np.abs(green - blue) <= .162 * chroma))
    samples = (rgb[height//10:height//2, width//9:8*width//9],
               rgb[height//2:3*height//4, :width//5],
               rgb[height//2:3*height//4, 4*width//5:])
    return np.concatenate([tile[::3, ::3].mean(axis=2).ravel() for tile in samples])


def motion(previous, current):
    if previous.dtype == bool:
        return float(np.count_nonzero(previous != current))
    return float(np.mean(np.abs(previous - current)))


class NominalMotion:
    """Render public geometry driven by the learner's own request tape."""

    def __init__(self, eye, settings):
        self.eye = eye
        self.arena = TT20Arena(**settings, view_mode=eye)
        self.arena.model.geom('ball').rgba[3] = 0.
        self.twin = CommandTwin(self.arena)
        self.camera = WRIST_CAMERA if eye == 'wrist' else CAMERAS[eye][0]

    def _sample(self):
        model, data = self.twin.model, self.twin.data
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        rgb = self.arena._render_view(self.arena._view_renderer(), data, self.camera)
        return motion_sample(self.eye, rgb)

    def reference(self, commands, start, stop):
        self.twin.reset()
        self.arena._color_roles()
        for command in commands[:start]:
            self.twin.advance(command)
        previous = self._sample()
        evidence = []
        for command in commands[start:stop]:
            self.twin.advance(command)
            sample = self._sample()
            evidence.append(motion(previous, sample))
            previous = sample
        return evidence

    def close(self):
        self.arena.close()


class PassiveDelay:
    """Accumulate recent, identifiable integer shifts without using point rewards."""

    def __init__(self, nominal):
        self.nominal = nominal
        self.delay = 0
        self.loss = np.zeros(MAX_DELAY + 1)

    def observe(self, commands, measured):
        start_time = perf_counter()
        active = np.flatnonzero(np.max(np.abs(commands[:len(measured)]), axis=1) > .025)
        result = dict(previous=self.delay, delay=self.delay, status='insufficient_motion', rendered=0)
        if len(active) >= 3:
            start = max(0, int(active[0]) - 1)
            stop = min(len(measured), int(active[-1]) + MAX_DELAY + 6)
            if stop - start > MAX_DELAY + 3:
                template = self.nominal.reference(commands, start, stop)
                measured = measured[start:stop]
                result.update(start=start, stop=stop, template=template, rendered=len(template) + 1)
                try:
                    match = identify_delay(template, measured, maximum=MAX_DELAY)
                except ValueError:
                    match = None
                if match is not None:
                    absolute = np.asarray(match['losses'])
                    delta = absolute - absolute.min()
                    result.update(match=match, status='ambiguous')
                    if absolute.min() < .72 and match['margin'] >= .025:
                        self.loss = .30 * self.loss + delta
                        best, other, *_ = np.argsort(self.loss)
                        if self.loss[other] - self.loss[best] >= .025:
                            self.delay = int(best)
                            result.update(delay=self.delay, status='accepted')
                        result['carried_loss'] = self.loss.tolist()
        result['wall_seconds'] = perf_counter() - start_time
        return result


class GameDelayPlayer(MatchPlayer):
    """Retain the usual serve and receive policies with a lead at command issuance."""

    def __init__(self, programs, catalogue, law, response, ranker_arena):
        banks = {'1': dict(programs=programs, trials=[]), '-1': dict(programs=[np.zeros(13)], trials=[])}
        super().__init__(banks, catalogue, law, response, 'catalog')
        self.player = LatencyPlayer(catalogue, law, response, 0, count=32)
        self.player.sim.close()
        self.player.sim = DrySimulator(law, arena=ranker_arena)

    @property
    def lead(self):
        return self.player.delay

    @lead.setter
    def lead(self, delay):
        self.player.delay = delay

    def command(self, tick, xyz, *, estimated_velocity=None):
        if self.role > 0 and self.phase == 'opening':
            tick += self.lead
        return super().command(tick, xyz, estimated_velocity=estimated_velocity)
