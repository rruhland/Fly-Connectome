"""Check capture of the active rival rate path and point accounting."""

import numpy as np
import pytest

from scripts.experimental.tt22_match import step_recorded, summarize


@pytest.mark.parametrize('trying', [False, True])
def test_live_record_keeps_the_rival_dispatch_and_restores_it(trying):
    class Arena:
        _rival_attempt = trying

        def _aim(self, *_):
            return self._command_to(-1) * 2.

        def _command_to(self, *_):
            return np.array((.1, -.2, .3, 0.))

        def step(self, action, *, render):
            rates = self._aim(-1) if trying else self._command_to(-1)
            self.used = rates.tolist()
            return None, -1, True, False, None

    arena = Arena()
    aim, direct = arena._aim, arena._command_to
    reward, done, cap, tape = step_recorded(arena, np.zeros(6))
    assert (reward, done, cap) == (-1, True, False)
    assert tape == arena.used and arena._aim == aim and arena._command_to == direct


def test_failed_service_is_kept_in_the_point_lengths_and_attempts():
    rows = [dict(role=1, longest=0, outcome=dict(legal_serve=False, legal_returns=0, rival_returns=0)),
            dict(role=-1, longest=10, outcome=dict(legal_serve=True, legal_returns=5, rival_returns=5))]
    metrics = summarize(rows)
    assert metrics['lengths'] == [0, 10]
    assert metrics['serve_attempts'] == {'pupil': 1, 'rival': 1}
    assert metrics['legal_serves'] == {'pupil': 0, 'rival': 1}
    assert metrics['legal_returns'] == {'pupil': 5, 'rival': 5}
