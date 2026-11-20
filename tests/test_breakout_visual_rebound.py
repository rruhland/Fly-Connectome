"""Exercise visual rebound estimation without playing native Breakout."""

import math
import json
from concurrent.futures import Future
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from breakout_brick_placement import OFFSETS, Shot
from breakout_visual_rebound import (GAINS, Rebound, VisualReboundPlacement, VisualReboundPolicy,
                                     VisualRebounds, centered_forecast)
from confirm_breakout_visual_rebound import matched_effects
import run_breakout_visual_rebound as pilot


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


def test_all_arms_get_same_arrival_forecast_when_no_shot_is_available():
    timing = SimpleNamespace(seconds=0.)
    placement = VisualReboundPlacement(timing, VisualReboundPolicy(), 18)
    for context in (None, SimpleNamespace(seconds=.5), SimpleNamespace(seconds=1.5)):
        assert placement.offset(context) == centered_forecast(context)


def test_pilot_aggregation_with_synthetic_life_receipts(monkeypatch, tmp_path):
    class LocalExecutor:
        def __init__(self, max_workers, mp_context):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def submit(self, task, *args):
            future = Future()
            future.set_result(task(*args))
            return future

    def mock_play(case, checkpoint, policy, duration, lag_frames):
        if policy is not None and policy.updates_enabled:
            policy.observe_rebound(Rebound(.4, .06, .54))
        score = int(policy is not None and policy.visual_updates > 0)
        return dict(case=pilot.asdict(case['episode']), board=case['board'],
                    layout=pilot.asdict(case['layout']), agent_seed=case['agent_seed'],
                    score=score, post_own_bricks=score, serve_bricks=0, own_opportunities=2,
                    brick_events=[dict(time_seconds=2., after_return=True)] if score else [],
                    own_returns=1, elapsed_seconds=duration, cameras=600, controller_started_seconds=.14,
                    life_lost=True, cleared=False, truncated=False, score_updates=0, shots=[],
                    visual_rebounds=[], matches=[])

    monkeypatch.setattr(pilot, 'ProcessPoolExecutor', LocalExecutor)
    monkeypatch.setattr(pilot, 'play', mock_play)
    monkeypatch.setattr(pilot.torch, 'set_num_threads', lambda threads: None)
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text('{}')
    output = tmp_path / 'pilot'
    pilot.run(output, checkpoint, 4, 4, 12.)
    report = json.loads((output / 'summary.json').read_text())
    spec = json.loads((output / 'spec.json').read_text())
    layouts = [tuple(map(tuple, row['layout']['cells'])) for rows in spec['phase_cases'].values()
               for row in rows]
    assert len(layouts) == len(set(layouts)) == 8
    assert report['matched_lifts']['visual_empty']['all'] == {
        'bricks_per_life': 1., 'bricks_per_minute': 5., 'return_fraction': 0.}
    assert report['test']['visual_trained']['initial_state'] == report['test']['visual_trained']['final_state']
    trained = json.loads((output / 'test-visual_trained.json').read_text())
    empty = json.loads((output / 'test-visual_empty.json').read_text())
    ordinary = matched_effects(trained, empty)['all']
    assert (ordinary['ahead'], ordinary['tied'], ordinary['behind']) == (4, 0, 0)
    assert ordinary['bricks_per_life_ci95'] == [1., 1.]
    assert ordinary['bricks_per_minute_ci95'] == [5., 5.]
