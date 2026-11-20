"""Exercise visual rebound estimation without playing native Breakout."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from breakout_brick_placement import OFFSETS, Shot
from breakout_visual_rebound import GAINS, Rebound, VisualReboundPlacement, VisualReboundPolicy, VisualRebounds


def camera_track(angle=.4, offset=.05):
    """Inject pixel quantization, uneven camera steps, two missing frames and a changed ball ID."""
    along = -.1
    for index in range(36):
        now = (12 * index // 5) / 120
        delta = now - .36
        paddle = [round(64 * (along - offset + .05 * delta)) / 64, -.453125]
        if delta < 0.:
            ball = [along + .092 * delta, -.4375 - .54 * delta]
        else:
            ball = [along + .55 * math.sin(angle) * delta, -.4375 + .55 * math.cos(angle) * delta]
        if abs(delta) < .027:
            yield now, [], None
        else:
            ball_id = 5 if delta < 0. else 12
            ball = [round(64 * point) / 64 for point in ball]
            yield now, [dict(id=2, observed=True, normalized_position=paddle),
                        dict(id=ball_id, observed=True, normalized_position=ball)], (2, ball_id)


@pytest.mark.parametrize('angle, offset', [(.4, .05), (-.45, -.06)])
def test_visual_estimator_recovers_signed_motion_from_quantized_track(angle, offset):
    model = VisualRebounds()
    recovered = [rebound for seconds, entities, pair in camera_track(angle, offset)
                 if (rebound := model.observe(seconds, entities, pair)) is not None]
    assert len(recovered) == 1
    rebound = recovered[0]
    assert rebound.seconds == pytest.approx(.36, abs=.04)
    assert rebound.angle == pytest.approx(angle, abs=.10)
    assert rebound.contact_offset == pytest.approx(offset, abs=.018)


def test_angular_evidence_calibrates_sparse_and_full_board_rays():
    policy = VisualReboundPolicy()
    for contact in (-.07, .045, -.055, .079, .06, -.075, .025, .055):
        policy.observe_rebound(Rebound(0., contact, -9. * contact))
    assert policy.visual_updates == 8 and policy.updates == 0
    assert GAINS[np.argmax(policy.probabilities)] == -9.
    assert policy.probabilities[GAINS < 0].sum() > .98

    # The learned negative slope gain needs positive paddle offset to target right.
    loaded = VisualReboundPolicy.from_state(policy.to_state(), updates_enabled=False)
    rng = np.random.default_rng(987)
    sparse = np.array([[.30, .15, .034]])
    full = np.array([[.30, .15, .034], [.36, .22, .034], [.27, .33, .034],
                     [.34, .39, .034], [.40, .31, .034], [.31, .26, .034]])
    for targets in (sparse, full):
        arm, coverage, _ = loaded.choose(0., -.44, targets, np.arange(len(OFFSETS)), rng)
        assert OFFSETS[arm] > 0. and coverage > .75
    loaded.observe_score(coverage, 1.)
    loaded.observe_rebound(Rebound(0., -.07, .1))
    assert loaded.to_state() == policy.to_state()


def test_only_verified_flight_receives_public_brick_reward():
    policy = VisualReboundPolicy()
    timing = SimpleNamespace(seconds=0.)
    placement = VisualReboundPlacement(timing, policy, 87)
    placement.shot = Shot(.1, .36, -.1, -.04, .8, 1)
    placement.likelihood = .9
    for seconds, entities, pair in camera_track():
        placement.observe(seconds, entities, pair, float(seconds < .15))
    assert len(placement.observed_rebounds) == 1
    assert placement.shot.score == 0.
    placement.observe(.75, [], None, 1., terminated=True)
    assert placement.shots[0]['credited'] and placement.shots[0]['score'] == 1.
    assert policy.visual_updates == 1 and policy.updates == policy.scored == 1
    assert policy.payoff_hits[1] > 7.04 and policy.execution_bias > 0.


def test_terminated_miss_does_not_train_score():
    policy = VisualReboundPolicy()
    placement = VisualReboundPlacement(SimpleNamespace(seconds=0.), policy, 18)
    placement.shot = Shot(.1, .36, 0., 0., .5, 3)
    placement.likelihood = .6
    placement.observe(.4, [], None, 0., terminated=True)
    assert not placement.shots[0]['credited']
    assert policy.updates == policy.visual_updates == 0
