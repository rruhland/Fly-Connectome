from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arcade_pursuit_credit import PursuitEvidence
from arcade_transfer_session import TransferSession


def terminate(session, intended, observed_motion, *, terminal=True, reward=0., delay=0,
              uncertain=False, changed_pair=False):
    evidence = PursuitEvidence()
    timing = SimpleNamespace(
        belief=SimpleNamespace(order=(2, 0, 1), states=True, probabilities=[.98], confirmed=0),
        candidate_dwell=int(uncertain), delay_commands=delay, camera_seconds=.02,
        state=lambda: session.timing,
    )
    for frame in range(15):
        time = frame * .02
        ball_id = 3 if changed_pair and frame >= 7 else 2
        detector = SimpleNamespace(
            canonical_entities=[
                dict(id=1, observed=True, normalized_position=[observed_motion * time, -.45]),
                dict(id=ball_id, observed=True, normalized_position=[.26, -.1 - time / 2]),
            ],
            pair_ids=(1, ball_id), binding=None,
        )
        timing.arrival_context = SimpleNamespace(intercept=.29, seconds=.70 - time)
        action = timing.belief.order[intended] if intended is not None else 0
        evidence.record(time, detector, timing, action, intended)
    session.life = SimpleNamespace(controller=SimpleNamespace(timing=timing, shot_memory=session.shot_memory),
                                   detector=detector, pursuit=evidence, seconds=.80, earned=reward,
                                   ended=True, terminated=terminal)
    session.end_life()
    return session.state()['pursuit_credit']


@pytest.mark.parametrize('intended,motion,options,expected', [
    (0, .25, {}, 'toward'),
    (2, .25, {}, None),
    (2, 0., {}, None),
    (2, -.25, {'delay': 8}, None),
    (2, -.25, {'uncertain': True}, None),
    (2, -.25, {'changed_pair': True}, None),
    (None, -.25, {}, None),
    (2, -.25, {'terminal': False}, None),
    (2, -.25, {'reward': 1.}, 'away'),
])
def test_short_life_without_credited_away_motion_preserves_pursuit(intended, motion, options, expected):
    session = TransferSession('fresh', 39, causal_pursuit=True)
    polarity = session.polarity
    credit = terminate(session, intended, motion, **options)
    assert session.polarity == polarity
    assert credit['failures'] == 0 and not credit['last']['revised']
    witness = credit['last']['witness']
    assert (None if witness is None else witness['direction']) == expected


def test_fresh_pursuit_revises_after_a_visibly_executed_wrong_way_chase():
    session = TransferSession('fresh', 39, causal_pursuit=True)
    polarity = session.polarity
    credit = terminate(session, 2, -.25)
    assert credit['last']['witness']['direction'] == 'away'
    assert credit['last']['revised'] and session.polarity == -polarity


def test_carried_relation_revises_after_two_supported_lives_and_state_restores():
    session = TransferSession('full', 39, causal_pursuit=True)
    polarity = session.polarity
    first = terminate(session, 2, -.25)
    assert first['failures'] == 1 and session.polarity == polarity
    checkpoint = session.state()
    session = TransferSession.from_state(checkpoint)
    assert session.state() == checkpoint
    second = terminate(session, 2, -.25)
    assert second['last']['revised'] and session.polarity == -polarity
    assert 'pursuit_credit' not in TransferSession('full', 39).state()
