import copy
import json
from pathlib import Path

import numpy as np

from report_pong_carried_learning import (
    costs, digest, load_receipts, paired_controls, rates, recovery, report_experiment, state_digest,
)


def write(path, data):
    path.write_text(json.dumps(data))


def make_experiment(tmp_path, games=36, missing=(), failed_plane_game=None, phase_start=0):
    root = tmp_path / 'run'
    directory = root / 'capped_beta'
    directory.mkdir(parents=True)
    source = tmp_path / 'package'
    source.mkdir()
    (source / 'pong.py').write_text('# frozen source\n')
    asset = tmp_path / 'asset.bin'
    asset.write_bytes(b'fixed prior')
    scripts = Path(__file__).parents[1] / 'scripts/experimental'
    assignments = [dict(index=i, phase='A_acquire', phase_index=i + phase_start,
        seed=i + 101, reverse=False, max_episode_ticks=2400,
        max_transition_cameras=1000) for i in range(games)]
    write(root / 'schedules.json', {'capped_beta': assignments})
    write(root / 'seed-exclusion-proof.json', {'canonical_ids': []})
    spec = dict(files={str(scripts / name): digest(scripts / name) for name in (
        'pong_carried_structure_learner.py', 'run_pong_carried_learning.py')},
        assets={str(asset): digest(asset)}, source={'pong.py': digest(source / 'pong.py')},
        schedules_sha256=digest(root / 'schedules.json'),
        exclusions_sha256=digest(root / 'seed-exclusion-proof.json'))
    spec['identity'] = state_digest(spec)
    write(root / 'spec.json', spec)
    state = dict(kind='capped_beta', lifetime_success=[0, 0], lifetime_failure=[0, 0],
        rng_state={'state': 123}, games=0, active_arm=None, updates_enabled=True)
    write(directory / 'checkpoint-initial.json', state)
    for assignment in assignments:
        i = assignment['index']
        if i in missing:
            continue
        before = copy.deepcopy(state)
        misses = int(i == failed_plane_game)
        hits = 5 - misses
        events = [dict(tick=2 * (j + 1), hit=j < hits, first=j == 0) for j in range(5)]
        state['lifetime_success'][0] += hits
        state['games'] += 1
        after = copy.deepcopy(state)
        tape = directory / f'game-{i:03d}-events.npz'
        np.savez_compressed(tape, physics_ticks=np.array([2] * 5),
            action=np.zeros(5, dtype=np.int8), physical_action=np.zeros(5, dtype=np.int8),
            elapsed_seconds=np.array([1 / 60] * 5), hits=np.array([int(e['hit']) for e in events]),
            opportunity_tick=np.array([e['tick'] for e in events]),
            opportunity_hit=np.array([e['hit'] for e in events]),
            opportunity_first=np.array([e['first'] for e in events]))
        row = dict(assignment=assignment, events=events, complete=True, arm=0,
            selection_mode='explore' if i < 16 else 'exploit',
            runtime_identity=spec['identity'], learner_before=before, learner_after=after,
            learner_before_sha256=state_digest(before), learner_after_sha256=state_digest(after),
            report=dict(physics_seconds=1 / 12, hits=hits, losses=0, wins=0, truncations=1),
            annotation=dict(own_opportunities=5, own_returns=hits), transitions=5,
            initial_cameras=1, transition_camera_upper_bound=5, initial_camera_upper_bound=1,
            wall_seconds=.1, streaming_audit=dict(passed=True),
            frozen_execution_parameters_checked=True, tape_file=str(tape), tape_sha256=digest(tape))
        write(directory / f'game-{i:03d}.json', row)
    write(directory / 'checkpoint-latest.json', state)
    write(directory / 'work-clock.json', dict(spec_sha256=digest(root / 'spec.json'), seconds=100))
    return root, source


def test_receipt_reader_excludes_miss_lists_checkpoints_and_summaries(tmp_path):
    write(tmp_path / 'game-000.json', {'assignment': {'index': 0}})
    write(tmp_path / 'game-000-miss.json', [{'action': 2}])
    write(tmp_path / 'checkpoint-latest.json', {'active_arm': None})
    write(tmp_path / 'summary.json', {'complete': True})
    assert load_receipts(tmp_path) == [{'assignment': {'index': 0}}]


def test_fixed_tail_is_scheduled_last32_not_latest_available32(tmp_path):
    root, source = make_experiment(tmp_path, missing=(33, 34, 35))
    report = report_experiment(root, source)
    stream = report['streams']['capped_beta']
    tail = stream['phases']['A_acquire']['fixed_tail']
    assert tail['scheduled_indices'] == list(range(4, 36))
    assert tail['metrics']['games'] == 29
    assert tail['conditions']['enough_later_opportunities']
    assert not tail['conditions']['exact_fixed_tail_complete']
    assert not tail['descriptive_window_conditions_met']
    assert not stream['complete'] and not report['qualification_declared']


def test_missed_plane_before_cap_counts_failure_without_invented_reward(tmp_path):
    root, source = make_experiment(tmp_path, failed_plane_game=35)
    report = report_experiment(root, source)
    stream = report['streams']['capped_beta']
    phase = stream['phases']['A_acquire']
    assert report['source_audit']['passed'] and stream['audit']['passed']
    assert phase['fixed_tail']['metrics']['later_returns'] == 127
    assert phase['fixed_tail']['metrics']['later_opportunities'] == 128
    assert phase['costs']['own_hits'] == 179
    assert phase['costs']['capped_games'] == 36
    assert phase['costs']['natural_ends'] == 0
    assert not report['qualification_declared']


def test_all_started_costs_and_rates_include_incomplete_exploratory_game():
    rows = [dict(events=[dict(hit=True, first=True)], complete=True,
        selection_mode='exploit'), dict(events=[dict(hit=False, first=True)],
        complete=False, selection_mode='explore')]
    assert rates(rows)['first_rate'] == .5
    assert rates(rows)['games'] == 2 and rates(rows)['completed_games'] == 1
    for row in rows:
        row.update(report=dict(physics_seconds=2., hits=1, losses=0, wins=0,
            truncations=0), transitions=100, initial_cameras=1,
            transition_camera_upper_bound=101, initial_camera_upper_bound=1,
            wall_seconds=3.)
    assert costs(rows)['physics_seconds'] == 4
    assert costs(rows)['transition_camera_upper_bound'] == 202
    assert costs(rows)['incomplete_games'] == 1


def test_recovery_uses_fixed25_bins_and_requires_confirmation():
    def events(hits):
        return [dict(hit=bool(hit), game=i + 1, phase_game=i + 1,
            phase_seconds=float(i + 1), phase_transition_cameras=50 * (i + 1),
            phase_initial_cameras=i + 1) for i, hit in enumerate(hits)]
    # One failed bin, two passing bins. No shifted rolling window may move the
    # declared confirmation before the end of bin3.
    result = recovery(events([False] * 25 + [True] * 50))
    assert result['confirmed'] and result['confirmation_opportunity'] == 75
    assert result['confirmation']['phase_seconds'] == 75
    assert result['first_passing_window_end']['phase_transition_cameras'] == 2500
    assert not recovery(events([True] * 49))['confirmed']


def test_discontinuous_checkpoint_and_source_tampering_are_flagged(tmp_path):
    root, source = make_experiment(tmp_path)
    path = root / 'capped_beta/game-020.json'
    row = json.loads(path.read_text())
    row['learner_before']['rng_state']['state'] = 999
    row['learner_before_sha256'] = state_digest(row['learner_before'])
    write(path, row)
    (source / 'pong.py').write_text('# changed source\n')
    report = report_experiment(root, source)
    assert not report['source_audit']['passed']
    audit = report['streams']['capped_beta']['audit']
    assert not audit['passed']
    assert 'game 20: broken carried-state boundary' in audit['problems']


def test_phase_boundaries_carry_evidence_and_reset_only_latency_clock(tmp_path):
    root, source = make_experiment(tmp_path)
    schedules = json.loads((root / 'schedules.json').read_text())
    for assignment in schedules['capped_beta']:
        i = assignment['index']
        assignment['phase'] = ('A_acquire', 'B_reverse', 'A_return')[i // 12]
        assignment['phase_index'] = i % 12
    write(root / 'schedules.json', schedules)
    spec = json.loads((root / 'spec.json').read_text())
    spec.pop('identity')
    spec['schedules_sha256'] = digest(root / 'schedules.json')
    spec['identity'] = state_digest(spec)
    write(root / 'spec.json', spec)
    directory = root / 'capped_beta'
    for assignment in schedules['capped_beta']:
        path = directory / f"game-{assignment['index']:03d}.json"
        row = json.loads(path.read_text())
        row['assignment'] = assignment
        row['runtime_identity'] = spec['identity']
        write(path, row)
    write(directory / 'work-clock.json', dict(spec_sha256=digest(root / 'spec.json'), seconds=100))
    result = report_experiment(root, source)['streams']['capped_beta']
    assert result['audit']['passed']
    boundaries = result['audit']['carried_phase_boundaries']
    assert len(boundaries) == 2
    assert [boundary['lifetime_successes_carried'] for boundary in boundaries] == [60, 120]
    assert all(boundary['exact_state_carried'] for boundary in boundaries)
    # Every phase has60 opportunities; confirmation occurs at its50th, using
    #10 of that phase's games rather than cumulative time since acquisition.
    for phase in result['phases'].values():
        assert phase['recovery']['confirmation_opportunity'] == 50
        assert phase['recovery']['confirmation']['phase_game'] == 10


def test_tail_only_control_has_fixed_metrics_but_no_onset_recovery_latency(tmp_path):
    root, source = make_experiment(tmp_path, games=32, phase_start=32)
    phase = report_experiment(root, source)['streams']['capped_beta']['phases']['A_acquire']
    assert phase['scheduled_phase_index_start'] == 32
    assert phase['fixed_tail']['conditions']['exact_fixed_tail_complete']
    assert phase['recovery']['confirmed'] is None


def test_paired_controls_join_explicit_live_indices_and_reject_wrong_starts():
    common = dict(phase='B_reverse', phase_index=32, seed=123, reverse=True,
        max_episode_ticks=2400, max_transition_cameras=1000)
    left_assignment = dict(common, index=96)
    right_assignment = dict(common, index=0, matched_live_index=96,
        matched_live_stream='replica0-live')
    left = dict(assignment=left_assignment, events=[dict(hit=True, first=True)],
        complete=True, report=dict(hits=1))
    right = dict(assignment=right_assignment, events=[dict(hit=False, first=True)],
        complete=True, report=dict(hits=0))
    schedules = {'replica0-live': [left_assignment], 'replica0-frozen': [right_assignment]}
    rows = {'replica0-live': [left], 'replica0-frozen': [right]}
    report = paired_controls(rows, schedules)['replica0-frozen']
    assert report['matching_valid']
    assert report['phases']['B_reverse']['paired_mean_own_hit_difference'] == 1
    right_assignment['seed'] = 999
    assert not paired_controls(rows, schedules)['replica0-frozen']['matching_valid']
