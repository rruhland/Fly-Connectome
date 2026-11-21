"""Fit an anonymous score-event attribution clock from a fresh receiver's own camera stream."""

import numpy as np

from scripts.experimental.tt3_role_discovery import (
    DiscoveryBinder,
    _anonymous_candidates,
    _relevance_contrasts,
    _predictions,
    fit_event_clock,
)


class OnlineDiscoveryBinder(DiscoveryBinder):
    """Give a player without Pong exposure the same event family and real target observations."""

    def __init__(self, model, seed):
        super().__init__(model, None, learn=True)
        self.bags = []
        self.pending = []
        self.score_events = 0
        self.rng = np.random.default_rng(seed)

    def _learn(self, tracks, reward=0.0):
        candidates = _anonymous_candidates(self._recent, self._image_shape)
        if len(candidates) >= 2:
            anonymous = [track for track, _ in candidates]
            bag = dict(
                clock="true" if reward else "neutral",
                ids=[t.id for t in anonymous],
                features=np.stack([row for _, row in candidates]),
            )
            self.bags.append(bag)
            if reward:
                self.score_events += 1
                if self.clock is None:
                    self.pending.append(
                        (
                            bag,
                            _relevance_contrasts(self._scene, self._recent, anonymous),
                        )
                    )
                if (
                    self.score_events in (3, 8, 16)
                    and len(self.bags) > self.score_events
                ):
                    self.clock = fit_event_clock(self.bags, "true", self.rng)
                    # Credit earlier anonymous observations once a first clock exists.
                    for old, differences in self.pending[:-1]:
                        _, chosen = _predictions(self.clock, [old])
                        self._bank.relevance.extend(differences[old["ids"][chosen[0]]])
                    self.pending.clear()
        super()._learn(tracks, reward if self.clock is not None else 0.0)
