"""Prospective schedules, reward-only selection and budget/barrier semantics."""
from concurrent.futures import Future
import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
import run_pong_relational_categorical_replication as acquisition


def test_entire_fixed_structural_family_and_independent_predrawn_controls():
    roster = acquisition.make_roster(np.array([.2, .3, .4, .5]))
    assert len(roster) == len({tuple(row['choices']) for row in roster}) == 16
    assert {row['choices'][0] for row in roster} == {2, 3}
    assert {row['choices'][3] for row in roster} == {0, 1}
    assert {row['choices'][4] for row in roster} == {-1, 1}
    assert all(row['coefficients'] == [0.] * 5 and not row['require_first_hit'] for row in roster)
    for seed in acquisition.ORDER_SEEDS:
        a = acquisition.draw_priority(roster, seed)
        assert a == acquisition.draw_priority(roster, seed)
        assert set(a['priority']) == {row['id'] for row in roster}
        assert a['initial_control'] in a['priority']


def test_both_new_train_and_duration_balanced_fresh_schedules_fixed_together():
    excluded = {1, 2, 3, acquisition.canonical_seed(acquisition.ROOT_SEED)}
    schedule = acquisition.fixed_schedules(excluded)
    assert schedule == acquisition.fixed_schedules(excluded)
    acquisition.validate_schedules(schedule, excluded)
    seeds = []
    for replica in ('0', '1'):
        assert len(schedule[replica]['train']['seeds']) == 16
        rows = schedule[replica]['fresh']['assignments']
        assert sum(row['seconds'] == 60 for row in rows) == 32
        assert sum(row['seconds'] == 20 for row in rows) == 32
        seeds += schedule[replica]['train']['seeds'] + [row['seed'] for row in rows]
    assert len({acquisition.canonical_seed(s) for s in seeds}) == 160
    bad = copy.deepcopy(schedule)
    bad['1']['train']['seeds'] = bad['0']['train']['seeds'].copy()
    with pytest.raises(ValueError): acquisition.validate_schedules(bad, excluded)
    bad = copy.deepcopy(schedule); bad['0']['fresh']['assignments'][0]['max_episode_ticks'] = 2400
    with pytest.raises(ValueError): acquisition.validate_schedules(bad, excluded)


def put(cache, policy, seeds, score, eligible=True):
    for seed in seeds:
        cache[acquisition.cache_key(policy, seed)] = dict(own_hits=score, fitness_eligible=eligible)


def test_same_complete_seed_totals_only_baseline_ties_and_initial_control_eligible():
    seeds = list(range(16)); cache = {}
    put(cache, acquisition.BASELINE, seeds, 3)
    put(cache, 'second', seeds, 3)
    put(cache, 'first', seeds[:8], 100)
    assert acquisition.select_endpoint(['first', 'second', 'control'], seeds, cache) == (acquisition.BASELINE, 48)
    put(cache, 'second', seeds, 4)
    put(cache, 'control', seeds, 5)
    assert acquisition.select_endpoint(['first', 'second', 'control'], seeds, cache) == ('control', 80)
    put(cache, 'first', seeds, 5)
    assert acquisition.select_endpoint(['first', 'second', 'control'], seeds, cache) == ('first', 80)
    priority = ['control', 'first', 'second']
    assert acquisition.rank_complete(priority, seeds, cache, priority)[:2] == [('control', 80), ('first', 80)]
    put(cache, 'first', [0], 999, eligible=False)
    assert acquisition.completed_score('first', seeds, cache) is None


def test_later_stage_ties_restore_fixed_priority_not_previous_score_order():
    cache = {}; seeds = list(range(8)); priority = ['first', 'second']
    put(cache, 'first', seeds[:4], 3); put(cache, 'second', seeds[:4], 4)
    stage4 = acquisition.rank_complete(priority, seeds[:4], cache, priority)
    assert [policy for policy, _ in stage4] == ['second', 'first']
    put(cache, 'first', seeds[4:], 5); put(cache, 'second', seeds[4:], 4)
    carried_order = [policy for policy, _ in stage4]
    stage8 = acquisition.rank_complete(carried_order, seeds, cache, priority)
    assert stage8 == [('first', 32), ('second', 32)]


class ImmediatePool:
    def submit(self, function, *args):
        future = Future()
        try: future.set_result(function(*args))
        except Exception as error: future.set_exception(error)
        return future


def test_dispatch_uses_compact_identity_and_charges_bad_receipt_without_fitness(tmp_path):
    folder = tmp_path / 'replica0'; folder.mkdir(); (folder / 'failures').mkdir()
    spec = dict(replicas={'0': {'folder': str(folder)}}, roster=[],
        workers_per_replica=4, game_camera_grant=1000,
        acquisition_max_episode_ticks=2400, identity='frozen')

    def key(model, seed, ticks, identity): return f'key{seed}'
    def evaluate(model, seed, ticks, directory, deadline, grant, identity):
        return dict(seed=seed, model_sha256=None, max_episode_ticks=ticks,
            runtime_identity=identity if seed == 1 else 'wrong', fitness_key=key(model, seed, ticks, identity),
            complete=True, fitness_eligible=True, own_hits=4, transitions=17,
            initial_cameras=1, transition_camera_upper_bound=17,
            initial_camera_upper_bound=1, receipt_file=f'{seed}.json')
    worker = SimpleNamespace(fitness_key=key, evaluate=evaluate)
    cache = {}; budget = acquisition.CameraBudget(2000)
    rows = acquisition.dispatch(ImmediatePool(), worker, [acquisition.BASELINE], [1, 2],
        cache, budget, spec, 0, time.monotonic() + 30, 'synthetic')
    assert len(rows) == 2 and budget.used == 1017 and budget.initials == 2
    assert budget.reserved == 0 and len(cache) == 1
    assert acquisition.completed_score(acquisition.BASELINE, [1, 2], cache) is None
    assert (folder / 'failures/key2.json').exists()


def test_endpoint_barrier_requires_both_complete_unchanged_acquisitions(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, 'DEST', tmp_path)
    monkeypatch.setattr(acquisition, 'verify_spec', lambda spec, approved: None)
    acquisition.write(tmp_path / 'spec.json', dict(schedules_sha256='prospectivelyfixed'))
    for replica in ('0', '1'):
        folder = tmp_path / f'replica{replica}'; folder.mkdir()
        endpoint = dict(acquisition_complete=True, learned_endpoint=False,
            model_path=None, model_sha256=None, replica=int(replica))
        acquisition.write(folder / 'endpoint.json', endpoint)
        if replica == '0':
            acquisition.write(folder / 'summary.json', dict(status='acquisition_complete',
                spec_sha256='approved', endpoint=endpoint))
    with pytest.raises(FileNotFoundError): acquisition.freeze_endpoints('approved')
    endpoint = acquisition.read(tmp_path / 'replica1/endpoint.json')
    acquisition.write(tmp_path / 'replica1/summary.json', dict(status='incomplete_no_learned_endpoint',
        spec_sha256='approved', endpoint=endpoint))
    with pytest.raises(ValueError): acquisition.freeze_endpoints('approved')
    acquisition.write(tmp_path / 'replica1/summary.json', dict(status='acquisition_complete',
        spec_sha256='approved', endpoint=endpoint))
    result = acquisition.freeze_endpoints('approved')
    assert result['both_endpoints_frozen'] and not result['validation_started']
    assert set(result['endpoints']) == {'0', '1'}
    with pytest.raises(FileExistsError): acquisition.freeze_endpoints('approved')
