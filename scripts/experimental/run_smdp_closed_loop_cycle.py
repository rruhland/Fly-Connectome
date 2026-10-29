"""Append one declared closed-loop TRAIN acquisition; reuse the pinned learner.

No simulator is imported or run. Existing source/helpers and candidate0 remain
unchanged. Collection and native evaluation belong to the evaluation worker.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
from run_smdp_adaptive_forest import sha, write, scenes, fit as pinned_fit
from pong_smdp_adaptive_forest import (STATE_DIM, state_features, observed_windows,
    observed_mc_returns, balanced_weights)

ROOT = Path('/workspace/m1b-experiments')
OLD = ROOT / 'pong-smdp-adaptive-forest-20261029-r1'
ADMISSION = ROOT / 'pong-smdp-native-20261029-r1/screen8'
COLLECTION = ROOT / 'pong-smdp-onpolicy-collection-20261029-r1'
DEST = ROOT / 'pong-smdp-closed-loop-refit-20261029-r1'
C0_SHA = 'ca99b349d1ea26a2036214e37d04e58a2d39a59cf26b44d5aeed7e18484df295'


def old_seed_map():
    manifest = json.loads((OLD / 'data-manifest.json').read_text())
    return {int(seed):int(start) for start, seed in manifest['start_id_to_seed'].items()}


def admitted_traces():
    paths = sorted(ADMISSION.glob('candidate_??.json'))
    if len(paths) != 8:
        raise ValueError('all eight existing candidate0 TRAIN traces must be admitted')
    for path in paths:
        receipt = json.loads(path.read_text())
        if receipt['model_sha256'] != C0_SHA:
            raise ValueError('candidate0 admission endpoint changed')
        records = receipt['records']; n = len(records)
        if n != receipt['transitions'] or not receipt['complete']:
            raise ValueError('admitted TRAIN evaluation trace must match its completed receipt')
        scene_path = Path(receipt['structured_feature_file'])
        visual = scenes(scene_path)
        if len(visual) != n:
            raise ValueError('declared admitted trace has N pre-action scenes')
        yield dict(name=path.stem, family='admitted_candidate0_TRAIN_evaluation', seed=receipt['seed'],
            ordinal=0, past_hit=False, scenes=visual,
            actions=np.array([r['action'] for r in records], np.int8),
            hits=np.array([r['hits'] for r in records], bool),
            natural=np.array([r['terminated'] for r in records], bool),
            truncated=np.array([r['truncated'] for r in records], bool),
            elapsed=np.array([r['elapsed_seconds'] for r in records]),
            ticks=np.array([r['physics_ticks'] for r in records]),
            baseline=np.argmax(np.array([r['q'] for r in records]), axis=1),
            files=[dict(path=str(p), sha256=sha(p)) for p in (path, scene_path)])


def validate_completed_collection():
    summary_path = COLLECTION / 'summary.json'
    if not summary_path.exists():
        raise RuntimeError('collection is still active or has no completed summary')
    summary = json.loads(summary_path.read_text())
    if (summary.get('source_frozen') is not True or summary.get('assets_frozen') is not True
            or summary.get('status') not in ('all64complete', 'bounded_partial')):
        raise ValueError('completed collection must verify frozen sources and assets')
    started = [r for r in summary['traces'] if r['initial_cameras'] > 0]
    paths = sorted((COLLECTION / 'traces').glob('site_??.json'))
    if (len(started) != summary['started'] or len(paths) != len(started)
            or {p.resolve() for p in paths} != {Path(r['receipt_file']).resolve() for r in started}):
        raise ValueError('ingested receipt set differs from every started collection trace')
    total = 0
    for entry in started:
        receipt_path = Path(entry['receipt_file'])
        if sha(receipt_path) != entry['receipt_sha256']:
            raise ValueError('collection receipt hash changed')
        receipt = json.loads(receipt_path.read_text())
        if receipt['model_sha256'] != C0_SHA or entry['model_sha256'] != C0_SHA:
            raise ValueError('collection receipt used another model')
        if (sha(receipt['scene_file']) != receipt['scene_sha256']
                or sha(receipt['transition_file']) != receipt['transition_sha256']):
            raise ValueError('collected scene or transition hash changed')
        if receipt['transitions'] != entry['transitions']:
            raise ValueError('collection receipt camera counts disagree')
        total += receipt['transitions']
    if total != summary['transition_cameras']:
        raise ValueError('started collection trace cameras do not match summary')
    return summary


def collection_traces():
    paths = sorted((COLLECTION / 'traces').glob('site_??.json'))
    for path in paths:
        receipt = json.loads(path.read_text())
        data_path = path.with_name(path.stem + '-transitions.npz')
        scene_path = path.with_name(path.stem + '-raw-scenes.npz')
        if not data_path.exists():
            if receipt.get('transitions', 0):
                raise ValueError('a collected trace with actual cameras has no transition file')
            continue
        with np.load(data_path, allow_pickle=False) as saved:
            data = {key:saved[key].copy() for key in saved.files}
        n = len(data['action'])
        if n != receipt['transitions']:
            raise ValueError('transition array length disagrees with completed receipt')
        if not n:
            continue
        visual = scenes(scene_path)
        if len(visual) != n + 1 or data['baseline_q'].shape != (n + 1, 3):
            raise ValueError('new closed-loop traces must retain N+1 scenes and baseline scores')
        if not np.array_equal(data['ordinal'], np.arange(n)):
            raise ValueError('public camera ordinal must count actual actions from zero')
        baseline = np.argmax(data['baseline_q'][:-1], axis=1)
        if 'baseline_actions' in data and not np.array_equal(data['baseline_actions'][:-1], baseline):
            raise ValueError('recorded baseline actions do not match actual baseline scores')
        decisions = data['option_decision'].astype(bool)
        selected = data['selected_option'][decisions].astype(int)
        greedy = data['greedy_option'][decisions].astype(int)
        probabilities = data['behavior_p'][decisions]
        # Before the first own hit, the baseline-only gate has probability one.
        past = np.r_[False, np.logical_or.accumulate(data['hits'].astype(bool))][:-1]
        post_decision = past[decisions]
        if (not np.isin(selected[post_decision], np.arange(4)).all()
                or not np.isin(greedy[post_decision], np.arange(4)).all()):
            raise ValueError('post-hit option decision has no declared option')
        expected = np.full((int(post_decision.sum()), 4), .0625)
        expected[np.arange(len(expected)), greedy[post_decision]] += .75
        if not np.array_equal(probabilities[post_decision], expected):
            raise ValueError('closed-loop option probabilities differ from fixed epsilon=.25')
        yield dict(name=path.stem, family='new_closed_loop_option_exploration', seed=int(receipt['seed']),
            ordinal=0, past_hit=False, scenes=visual, actions=data['action'], hits=data['hits'].astype(bool),
            natural=data['terminated'].astype(bool), truncated=data['truncated'].astype(bool),
            elapsed=data['elapsed_seconds'], ticks=data['physics_ticks'], baseline=baseline,
            behavior=dict(post_hit_option_decisions=int(post_decision.sum()),
                selected_counts=np.bincount(selected[post_decision], minlength=4).tolist(),
                fixed_probabilities_verified=True),
            files=[dict(path=str(p), sha256=sha(p)) for p in (path, data_path, scene_path)])


class Appender:
    def __init__(self, old_data, seed_map):
        self.old = old_data
        self.seed_map = dict(seed_map)
        self.scales = old_data['scales']
        self.state_chunks = [old_data['states']]
        self.valid_chunks = [old_data['state_valid']]
        self.start_chunks = [old_data['state_start']]
        self.state_count = len(old_data['states'])
        self.transitions = {key:[old_data[key]] for key in
            ('source', 'successor', 'option', 'elapsed', 'kind', 'start', 'cameras', 'mc')}
        self.receipts = []

    def add(self, trace):
        n = len(trace['actions'])
        if not n:
            raise ValueError('nonempty actual trace required')
        if not np.allclose(trace['elapsed'], np.asarray(trace['ticks']) / 120, rtol=0, atol=1e-14):
            raise ValueError('actual reported sensor tick durations disagree')
        if not np.isin(trace['hits'], [0, 1]).all():
            raise ValueError('actual own-hit events must be binary camera observations')
        seed = int(trace['seed']); start = self.seed_map.setdefault(seed, len(self.seed_map))
        prior = np.r_[bool(trace['past_hit']), np.logical_or.accumulate(trace['hits']) | bool(trace['past_hit'])]
        vectors = []; valid = []
        for i, scene in enumerate(trace['scenes']):
            vector = state_features(scene, (trace['ordinal'] + i) % 5, prior[i], self.scales)
            vectors.append(np.zeros(STATE_DIM, np.float32) if vector is None else vector)
            valid.append(vector is not None)
        available = np.arange(n) + 1 < len(vectors)
        windows, rejected = observed_windows(trace['actions'], trace['hits'], trace['natural'], trace['baseline'], available)
        mc = observed_mc_returns(trace['hits'], trace['natural'], trace['elapsed'])
        cumulative = np.r_[0., np.cumsum(trace['elapsed'])]
        local = {key:[] for key in self.transitions}
        for source, option, end, kind in windows:
            local['source'].append(self.state_count + source)
            local['successor'].append(self.state_count + end + 1 if end + 1 < len(vectors) else -1)
            local['option'].append(option); local['elapsed'].append(cumulative[end + 1] - cumulative[source])
            local['kind'].append(kind); local['start'].append(start)
            local['cameras'].append(end - source + 1); local['mc'].append(mc[source])
        for key in self.transitions:
            self.transitions[key].append(np.asarray(local[key], dtype=self.old[key].dtype))
        self.state_chunks.append(np.asarray(vectors, np.float32)); self.valid_chunks.append(np.asarray(valid, bool))
        self.start_chunks.append(np.full(len(vectors), start, np.int32)); self.state_count += len(vectors)
        self.receipts.append(dict(name=trace['name'], family=trace['family'], seed=seed, start_id=start,
            source_cameras=n, saved_scenes=len(vectors), own_hit_events=int(np.sum(trace['hits'])),
            natural_end_events=int(np.sum(trace['natural'])), smdp_rows=len(windows),
            option_counts=np.bincount(local['option'], minlength=4).tolist(), rejected=rejected,
            behavior=trace.get('behavior'), files=trace.get('files', [])))

    def finish(self):
        data = {key:np.concatenate(chunks) for key, chunks in self.transitions.items()}
        data.update(states=np.concatenate(self.state_chunks), state_valid=np.concatenate(self.valid_chunks),
            state_start=np.concatenate(self.start_chunks), scales=self.scales.copy())
        return data


def prepare(output):
    if output.exists():
        raise FileExistsError('one declared collection/refit cycle only')
    output.mkdir(parents=True)
    seed_map = old_seed_map()
    admission = list(admitted_traces())
    assert sum(len(t['actions']) for t in admission) == 3150
    assert all(t['seed'] in seed_map for t in admission)
    plan = dict(method='one closed-loop collect then same observed-MC-initialized adaptive SMDP-FQI',
        candidate0=str(OLD / 'final-forest.npz'), candidate0_sha256=C0_SHA,
        old_data=str(OLD / 'smdp-data.npz'), old_data_sha256=sha(OLD / 'smdp-data.npz'),
        old_spec_sha256=sha(OLD / 'spec.json'), old_distinct_starts=len(seed_map),
        admitted_all_candidate0_TRAIN_evaluation=True, admission_trace_count=8,
        admission_cameras=3150, admission_hits=sum(int(t['hits'].sum()) for t in admission),
        admission_files=[f for trace in admission for f in trace['files']],
        collection_root=str(COLLECTION), planned_starts=64, phase_counts=[32, 32],
        exclude_old_and_admitted_starts=True, collection_seconds=180, collection_camera_cap=64000,
        collection_initial_cap=64, collector='frozen c0, baseline until actual hit, then .75greedy+.25uniform4options',
        greedy_option_probability=.8125, other_option_probability=.0625,
        command_option='16 constant command cameras or earlier actual hit/naturalend', baseline_option='one40069camera',
        ingestion='all valid observed constant-command windows plus actual baseline1steps; no changing-action relabel',
        mc_unknown_tail='omit only from MC initialization; retain valid transition rows for FQI',
        weights='same global equal option mass then equal distinct start-option mass then uniform windows',
        learned_state_unchanged=True, estimator_and_runtime_unchanged=True,
        refit='one observed BEHAVIOR-tail MC initialization plus up to32FQI;180s allcharged;lastcompleteonly',
        no_new_native_in_learner=True, no_fresh_data=True, fresh63233_remains_unseen=True,
        no_hyperparameter_sweep=True,
        collection_completion_guard='completed frozen-source/assets summary; every started receipt and camera; model/receipt/scene/transition hashes',
        source_files=[dict(path=str(Path(__file__)), sha256=sha(__file__)),
            *json.loads((OLD / 'spec.json').read_text())['source_files']])
    assert sha(OLD / 'final-forest.npz') == C0_SHA
    write(output / 'cycle-plan.json', plan)
    print(json.dumps(dict(status='prepared_no_collection_or_fit', plan_sha256=sha(output / 'cycle-plan.json'))))


def append(output, approved_plan, approved_collection):
    began = time.monotonic()
    if approved_plan != sha(output / 'cycle-plan.json'):
        raise ValueError('parent-approved fixed cycle plan required')
    plan = json.loads((output / 'cycle-plan.json').read_text())
    if approved_collection != sha(COLLECTION / 'spec.json'):
        raise ValueError('parent-approved collection spec required')
    if (output / 'smdp-data.npz').exists():
        raise FileExistsError('the declared append operation cannot silently repeat')
    assert all(sha(f['path']) == f['sha256'] for f in plan['source_files'])
    assert sha(OLD / 'smdp-data.npz') == plan['old_data_sha256']
    assert all(sha(f['path']) == f['sha256'] for f in plan['admission_files'])
    collection_summary = validate_completed_collection()
    with np.load(OLD / 'smdp-data.npz', allow_pickle=False) as saved:
        old = {key:saved[key].copy() for key in saved.files}
    appender = Appender(old, old_seed_map())
    admission = list(admitted_traces()); collected = list(collection_traces())
    if (len(collected) != collection_summary['started']
            or sum(len(t['actions']) for t in collected) != collection_summary['transition_cameras']):
        raise ValueError('not every started collection trace and camera was ingested')
    collected_seeds = {t['seed'] for t in collected}
    if len(collected_seeds) != len(collected) or collected_seeds & set(appender.seed_map):
        raise ValueError('new collected TRAIN starts must be distinct and exclude all old/admitted starts')
    for trace in admission + collected:
        appender.add(trace)
    data = appender.finish()
    eligible = data['state_valid'][data['source']]
    known = eligible & np.isfinite(data['mc'])
    weights = balanced_weights(data['option'], data['start'], eligible)
    mc_weights = balanced_weights(data['option'], data['start'], known)
    per_option = []
    for option in range(4):
        rows = data['option'] == option; valid = rows & eligible
        per_option.append(dict(option=option, actual_rows=int(rows.sum()), eligible_rows=int(valid.sum()),
            distinct_starts=int(len(np.unique(data['start'][valid]))), known_mc_rows=int((rows & known).sum()),
            unknown_mc_rows=int((valid & ~np.isfinite(data['mc'])).sum()),
            hit_windows=int((rows & (data['kind'] == 1)).sum()), natural_end_windows=int((rows & (data['kind'] == 2)).sum()),
            fqi_weight_mass=float(weights[rows].sum()), mc_weight_mass=float(mc_weights[rows].sum())))
    np.savez_compressed(output / 'smdp-data.npz', **data)
    spec = json.loads((OLD / 'spec.json').read_text())
    spec.update(data_sha256=sha(output / 'smdp-data.npz'), source_files=plan['source_files'],
        cycle=dict(plan_sha256=approved_plan, collection_spec_sha256=approved_collection,
            admission='all8 existing candidate0 TRAIN evaluation trajectories explicitly admitted',
            old_data_sha256=plan['old_data_sha256'], collector_model_sha256=C0_SHA,
            no_initial_policy_replacement=True))
    write(output / 'spec.json', spec)
    manifest = dict(old_manifest_path=str(OLD / 'data-manifest.json'), old_manifest_sha256=sha(OLD / 'data-manifest.json'),
        collection_summary_path=str(COLLECTION / 'summary.json'), collection_summary_sha256=sha(COLLECTION / 'summary.json'),
        added_traces=appender.receipts, start_id_to_seed={value:key for key,value in appender.seed_map.items()})
    write(output / 'data-manifest.json', manifest)
    summary = dict(status='appended_no_fit', old_rows=len(old['source']), combined_rows=len(data['source']),
        eligible_rows=int(eligible.sum()), combined_distinct_starts=len(appender.seed_map), by_option=per_option,
        admission_traces=8, admission_cameras=3150, collection_traces=len(collected),
        collection_cameras=sum(len(t['actions']) for t in collected),
        collection_actual_hit_events=sum(int(t['hits'].sum()) for t in collected),
        artifact_data_bytes=(output / 'smdp-data.npz').stat().st_size,
        elapsed_seconds=time.monotonic() - began, spec_sha256=sha(output / 'spec.json'))
    write(output / 'inventory.json', summary)
    print(json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'append', 'fit'])
    parser.add_argument('--output', type=Path, default=DEST)
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--approved-collection-spec-sha256')
    parser.add_argument('--approved-spec-sha256')
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args.output)
    elif args.operation == 'append':
        append(args.output, args.approved_plan_sha256, args.approved_collection_spec_sha256)
    else:
        pinned_fit(args.output, args.approved_spec_sha256)


if __name__ == '__main__':
    main()
