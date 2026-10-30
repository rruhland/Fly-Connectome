import copy
import json

import numpy as np
import pytest

import run_pong_carried_qualification as qualification
from pong_carried_structure_learner import CarriedStructureLearner, state_hash


def test_prospective_wall_budgets_cover_worst_case_cameras_with_margin():
    assert qualification.wall_budget(192000) == 2760
    assert qualification.wall_budget(240000) == 3480
    assert qualification.wall_budget(96000) == 1380
    assert qualification.wall_budget(64000) == 960


def prepare_without_native(tmp_path, monkeypatch, replicas=2, games_per_phase=64):
    model = tmp_path / 'assets'
    model.mkdir()
    (model / 'rbf-model.npz').write_bytes(b'frozen-test-placeholder')
    (model / 'radial-cap.json').write_text('{}')
    calibration = model / 'calibration.json'
    calibration.write_text('{}')
    monkeypatch.setattr(qualification, 'MODEL', model)
    monkeypatch.setattr(qualification, 'EXPECTED', qualification.sha(model / 'rbf-model.npz'))
    monkeypatch.setattr(qualification, 'CALIBRATION', calibration)
    monkeypatch.setattr(qualification, 'load_calibration_scales', lambda path: np.ones(4))
    def exclusions(directory):
        qualification.write(directory / 'seed-exclusion-proof.json', dict(canonical_ids=[1, 2, 3]))
        return {1, 2, 3}
    monkeypatch.setattr(qualification, 'exclusions', exclusions)
    def forbidden(*args, **kwargs):
        raise AssertionError('preparation attempted a native game')
    monkeypatch.setattr(qualification, 'evaluate_game', forbidden)
    destination = tmp_path / 'prepared'
    result = qualification.prepare(destination, algorithm='capped_beta',
        replicas=replicas, games_per_phase=games_per_phase)
    return destination, result


@pytest.mark.parametrize('replicas,games,total', [(2, 64, 544000), (3, 80, 880000)])
def test_replication_starts_are_new_and_controls_match_only_fixed_tails(
        tmp_path, monkeypatch, replicas, games, total):
    destination, result = prepare_without_native(tmp_path, monkeypatch, replicas, games)
    assert result['status'] == 'prepared_no_native'
    assert result['maximum_transition_cameras'] == total
    spec = json.loads((destination / 'spec.json').read_text())
    schedules = json.loads((destination / 'schedules.json').read_text())
    assert qualification.validate_schedules(schedules, spec['streams'], games, {1, 2, 3}) == replicas * games * 3
    assert len(schedules['replica0-reset']) == 96
    assert len(schedules['replica0-frozen']) == 64
    for name in ('replica0-reset', 'replica0-frozen'):
        for row in schedules[name]:
            live = schedules[row['matched_live_stream']][row['matched_live_index']]
            assert row['seed'] == live['seed'] and row['reverse'] == live['reverse']
            assert row['phase_index'] >= games - 32
            assert row['control_prefix_omitted']
    broken = copy.deepcopy(schedules)
    broken['replica0-frozen'][0]['seed'] += 100
    with pytest.raises(ValueError, match='paired control'):
        qualification.validate_schedules(broken, spec['streams'], games, {1, 2, 3})


def annotation(hits=6, opportunities=6):
    return dict(own_opportunities=opportunities, own_returns=hits,
        first_incoming=dict(success=hits > 0), maximum_consecutive_returns=hits,
        seconds=20., opponent_opportunities=6, opponent_returns=6,
        unfinished_incoming_approach=False, natural_serve_win=False,
        post_own_opponent_opportunities=5, post_own_opponent_returns=5)


def test_tail_acceptance_includes_failed_exploration_and_requires_full_exposure():
    assignments = [dict(index=i, phase='B_reverse', phase_index=32 + i) for i in range(32)]
    rows = [dict(assignment=row, complete=True, annotation=annotation(), arm=4,
        report=dict(hits=6), selection_mode='exploit') for row in assignments]
    metric = qualification.phase_metrics(rows, assignments, 64)['B_reverse']['last32']
    assert metric['passes'] and metric['later_opportunities'] == 160
    failed = copy.deepcopy(rows)
    for row in failed[:2]:
        row.update(annotation=annotation(0, 1), report=dict(hits=0),
            selection_mode='surprise_sweep')
    metric = qualification.phase_metrics(failed, assignments, 64)['B_reverse']['last32']
    assert not metric['passes'] and metric['first_incoming_returns'] == 30
    partial = qualification.phase_metrics(rows[:-1], assignments, 64)['B_reverse']['last32']
    assert not partial['passes'] and partial['completed_tail_games'] == 31


def test_tail_acceptance_rejects_too_few_later_opportunities():
    assignments = [dict(index=i, phase='A_return', phase_index=32 + i) for i in range(32)]
    rows = [dict(assignment=row, complete=True, annotation=annotation(3, 3), arm=4,
        report=dict(hits=3), selection_mode='exploit') for row in assignments]
    metric = qualification.phase_metrics(rows, assignments, 64)['A_return']['last32']
    assert metric['first_incoming_return_fraction'] == metric['later_return_fraction'] == 1.
    assert metric['later_opportunities'] == 64 and not metric['passes']


def test_stream_has_its_own_fixed_clock_and_stops_after_partial_game(tmp_path, monkeypatch):
    assignments = [dict(index=i, phase='A_acquire', phase_index=i,
        seed=10 + i, reverse=False, max_episode_ticks=2400,
        max_transition_cameras=1000) for i in range(3)]
    qualification.write(tmp_path / 'schedules.json', {'live': assignments})
    spec = dict(streams={'live': dict(role='live', kind='discounted_ucb',
        seconds=10., random_seed=1)}, roster=[{}, {}], evidence_cap=8.,
        discount=.95, exploration=.15, games_per_phase=64, identity='test-runtime')
    monkeypatch.setattr(qualification, 'verify_spec', lambda *args: spec)
    monkeypatch.setattr(qualification, 'seed_rngs', lambda: None)
    monkeypatch.setattr(qualification.base, 'initialize', lambda: None)
    monkeypatch.setattr(qualification, 'load_readout', lambda *args: object())
    clock = [100.]
    monkeypatch.setattr(qualification.time, 'monotonic', lambda: clock[0])
    deadlines = []
    def game(baseline, learner, shadow, roster, assignment, directory, deadline, identity):
        deadlines.append(deadline)
        complete = clock[0] + 6. <= deadline
        clock[0] = min(clock[0] + 6., deadline)
        for agent in (learner, shadow):
            agent.begin_game()
            agent.observe(own_hits=1, reward=0., terminated=False,
                truncated=complete, elapsed_seconds=.02)
        return dict(assignment=assignment, complete=complete, arm=0,
            selection_mode='initial_sweep', report=dict(hits=1),
            annotation=annotation(1, 1), transitions=10, initial_cameras=1,
            transition_camera_upper_bound=10, initial_camera_upper_bound=1,
            streaming_audit=dict(passed=complete))
    monkeypatch.setattr(qualification, 'evaluate_game', game)
    summary = qualification.run_stream(tmp_path, 'approved', 'live')
    assert deadlines == [110., 110.]
    assert not summary['complete'] and summary['completed_games'] == 1
    assert summary['scheduled_games'] == 3 and summary['wall_seconds'] == 10.
    assert summary['transitions'] == 20


def test_frozen_control_uses_closed_actual_last_A_arm_and_checks_source_identity(tmp_path):
    folder = tmp_path / 'replica0-live'
    folder.mkdir()
    acquired = CarriedStructureLearner('discounted_ucb', arms=2)
    acquired.games = 64
    acquired.success = [0., 8.]
    acquired.failure = [4., 0.]
    assert acquired.best_arm() == 1
    checkpoint = folder / 'checkpoint-end-A.json'
    qualification.write(checkpoint, acquired.state())
    receipt = dict(complete=True, runtime_identity='pinned-runtime', arm=0,
        assignment=dict(phase='A_acquire', phase_index=63),
        learner_after_sha256=state_hash(acquired.state()))
    receipt_path = folder / 'game-063.json'
    qualification.write(receipt_path, receipt)
    ready_path = folder / 'checkpoint-end-A-ready.json'
    def ready():
        qualification.write(ready_path, dict(stream='replica0-live',
            runtime_identity='pinned-runtime', spec_sha256='approved',
            checkpoint_sha256=qualification.sha(checkpoint),
            receipt_sha256=qualification.sha(receipt_path)))
    ready()
    frozen, manifest = qualification.frozen_prefix(tmp_path, 'replica0-live',
        dict(identity='pinned-runtime', games_per_phase=64), 'approved')
    assert frozen.incumbent == manifest['frozen_arm'] == receipt['arm'] == 0
    assert frozen.kind == 'frozen' and not frozen.updates_enabled
    assert manifest['final_acquisition_receipt_sha256'] == qualification.sha(receipt_path)
    receipt['runtime_identity'] = 'foreign-runtime'
    qualification.write(receipt_path, receipt)
    ready()
    with pytest.raises(ValueError, match='actual final acquired'):
        qualification.frozen_prefix(tmp_path, 'replica0-live',
            dict(identity='pinned-runtime', games_per_phase=64), 'approved')


def test_control_attribution_requires_each_predeclared_twenty_point_lift():
    def stream(a, b, returned):
        return dict(complete=True, metrics={phase: dict(last32=dict(return_fraction=rate, passes=True))
            for phase, rate in zip(qualification.PHASES, (a, b, returned))})
    results = {'replica0-live': stream(.99, .98, .99),
        'replica0-reset': stream(.70, .70, .70),
        'replica0-frozen': stream(None, .10, .99)}
    assert qualification.control_attribution(results)['passes']
    results['replica0-reset'] = stream(.90, .70, .70)
    assert not qualification.control_attribution(results)['passes']
    results['replica0-reset'] = stream(.70, .70, .70)
    results['replica0-frozen']['metrics']['A_return']['last32']['passes'] = False
    assert not qualification.control_attribution(results)['passes']
    results['replica0-reset'] = stream(None, .70, .70)
    assert not qualification.control_attribution(results)['passes']
