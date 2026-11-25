"""Opt-in causal high-rate RGB observer; standard optics remain the fallback.

The host supplies a camera renderer and clock. No live physical state enters
the estimator or the actor. The unchanged nominal body model uses own encoders.
"""
from time import process_time

import numpy as np

from .tt75_retina import Retina
from .tt75_dense_fit import DenseFlight
from .tt83_observer_adapter import PublicTracker


class ExposureSession:
    def __init__(self, arena, rig, wide, settings):
        self.arena = arena
        self.retina = Retina(arena, rig, wide)
        self.flight = DenseFlight(self.retina.period, settings)
        self.original = arena._contacts
        self.next_image = 0.
        self.fit_cpu = 0.
        self.fit_calls = 0
        arena._contacts = self.contacts

    def begin(self):
        self.retina.reset()
        self.flight.reset()
        self.next_image = 0.
        self.capture(0.)

    def capture(self, now):
        point, _, _ = self.retina.capture(now)
        self.flight.ingest(now, point)
        self.next_image += self.retina.period

    def contacts(self, incoming):
        self.original(incoming)
        now = float(self.arena.data.time)
        if now >= self.next_image - 1.e-8 and not (self.arena.terminated or self.arena.truncated):
            self.capture(now)

    def fit(self, now):
        started = process_time()
        answer = self.flight.fit(now)
        self.fit_cpu += process_time() - started
        self.fit_calls += 1
        return answer

    def stats(self):
        r = self.retina
        return dict(camera_calls=r.calls, wide_calls=r.wide_calls,
                    valid_stereo=r.valid, camera_cpu_seconds=r.cpu,
                    fit_calls=self.fit_calls, fit_cpu_seconds=self.fit_cpu)

    def close(self):
        self.arena._contacts = self.original
        self.retina.close()


def actor_class(parent):
    class Observed(parent):
        def begin(self):
            super().begin()
            self.public = None
            self.observer_inbound = False
            self.observer_decisions = []
            if self.observation_enabled:
                self.exposures.begin()

        def step(self, tick, eye, x, q, reference, base):
            if not self.observation_enabled:
                return super().step(tick, eye, x, q, reference, base)
            if self.public is None:
                self.public = PublicTracker(eye)
            selected = self.public.update(tick * self.engine.twin.control_dt, x,
                                          self.exposures.fit(tick * self.engine.twin.control_dt))
            dense = self.public.dense is not None
            if selected is not None and selected[4] < -.9 and selected[1] < -.90:
                self.observer_inbound = True
            # The direct field and temporal field receive the same selected
            # observation as search. No precomputed old-observation base leaks
            # into a dense decision. On fallback retain the exact stock base.
            if dense:
                home = self.observer_home
                base = home.copy()
                if selected[4] < -.9 or (self.observer_inbound and selected[1] < .50):
                    target, confidence = self.observer_raw.get(selected, q)
                    if target is not None:
                        base = home + confidence * (target - home)
            self.observer_decisions.append(dict(tick=tick, source='dense' if dense else 'legacy',
                x=None if selected is None else selected.tolist(),
                legacy=None if x is None else np.asarray(x).tolist(),
                dense_bounced=None if not dense else self.public.latest.bounced))
            return super().step(tick, self.public, selected, q, reference, base)

    return Observed
