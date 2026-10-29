"""One second feedback append and exactly optimized, otherwise unchanged refit."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import run_smdp_closed_loop_cycle as ingestion
from run_smdp_adaptive_forest import sha, write, scenes
from pong_smdp_adaptive_forest import balanced_weights
from run_smdp_fast_fit import fit as fast_fit

ROOT = Path('/workspace/m1b-experiments')
PRIOR = ROOT / 'pong-smdp-closed-loop-refit-20261029-r1'
COMPARISON = ROOT / 'pong-smdp-cycle-comparison-20261029-r1'
COLLECTION = ROOT / 'pong-smdp-onpolicy-collection-20261029-r2'
DEST = ROOT / 'pong-smdp-feedback2-refit-20261029-r1'
C0_SHA = 'ca99b349d1ea26a2036214e37d04e58a2d39a59cf26b44d5aeed7e18484df295'
C1_SHA = 'e58a73f8ba9fe695aa7e1815fc268aaf43bef0759f8a65760fa58bcf8009f2cd'


def prior_seeds():
    manifest = json.loads((PRIOR / 'data-manifest.json').read_text())
    return {int(seed):int(start) for start, seed in manifest['start_id_to_seed'].items()}


def comparison_traces():
    summary = json.loads((COMPARISON / 'summary.json').read_text())
    if summary['source_frozen'] is not True or summary['assets_frozen'] is not True:
        raise ValueError('completed comparison must verify frozen sources and assets')
    groups = [('matched32/c0', 24, C0_SHA), ('c1-screen8', 8, C1_SHA), ('matched32/c1', 24, C1_SHA)]
    camera_sum = 0; paths_seen = set()
    old_admissions = json.loads((PRIOR / 'data-manifest.json').read_text())['added_traces']
    old_receipts = {f['path'] for r in old_admissions for f in r['files'] if f['path'].endswith('.json')}
    for folder, expected, model_sha in groups:
        paths = sorted((COMPARISON / folder).glob('candidate_??.json'))
        if len(paths) != expected:
            raise ValueError('must admit all 24 new c0 and all 32 c1 comparison traces')
        for path in paths:
            if str(path) in old_receipts or path in paths_seen:
                raise ValueError('cached c0 eight or duplicate receipt would be admitted twice')
            paths_seen.add(path)
            receipt = json.loads(path.read_text()); records = receipt['records']; n = len(records)
            if receipt['model_sha256'] != model_sha or not receipt['complete'] or n != receipt['transitions']:
                raise ValueError('comparison admission model/count/completion mismatch')
            scene_path = Path(receipt['structured_feature_file']); visual = scenes(scene_path)
            if len(visual) != n:
                raise ValueError('comparison admission must have its actual N pre-action scenes')
            camera_sum += n
            yield dict(name=folder + '/' + path.stem, family='admitted_new_comparison_' + ('c0' if model_sha == C0_SHA else 'c1'),
                seed=receipt['seed'], ordinal=0, past_hit=False, scenes=visual,
                actions=np.array([r['action'] for r in records], np.int8),
                hits=np.array([r['hits'] for r in records], bool),
                natural=np.array([r['terminated'] for r in records], bool),
                truncated=np.array([r['truncated'] for r in records], bool),
                elapsed=np.array([r['elapsed_seconds'] for r in records]), ticks=np.array([r['physics_ticks'] for r in records]),
                baseline=np.argmax(np.array([r['q'] for r in records]), axis=1),
                files=[dict(path=str(p), sha256=sha(p)) for p in (path, scene_path)])
    if camera_sum != summary['new_transition_cameras'] or len(paths_seen) != 56:
        raise ValueError('admitted comparison cameras do not cover all 56 new receipts')


def source_files():
    prior = json.loads((PRIOR / 'spec.json').read_text())['source_files']
    assert all(sha(entry['path']) == entry['sha256'] for entry in prior), 'prior source changed'
    paths = [Path(entry['path']) for entry in prior]
    paths += [Path(__file__), Path(__file__).with_name('run_smdp_fast_fit.py'),
        Path(__file__).with_name('pong_smdp_fast_training.py')]
    return [dict(path=str(path), sha256=sha(path)) for path in dict.fromkeys(paths)]


def prepare(output):
    if output.exists():
        raise FileExistsError('one declared second feedback cycle only')
    output.mkdir(parents=True)
    admitted = list(comparison_traces()); old = prior_seeds()
    excluded = set(old) | {t['seed'] for t in admitted}
    assert len(old) == 139 and len(excluded) == 142
    parity = PRIOR / 'training-speed-parity.json'
    proof = json.loads(parity.read_text())
    if not all(proof[key] is True for key in ('presence_exact', 'observed_predictions_exact',
            'next_scores_support_exact', 'bellman_targets_exact', 'no_fits', 'no_native')):
        raise ValueError('exact query/target optimization proof is missing')
    plan = dict(method='one second feedback acquisition then unchanged MC+FQI with exact faster queries',
        acting_model=str(PRIOR / 'final-forest.npz'), acting_model_sha256=C1_SHA,
        prior_data=str(PRIOR / 'smdp-data.npz'), prior_data_sha256=sha(PRIOR / 'smdp-data.npz'),
        prior_spec_sha256=sha(PRIOR / 'spec.json'), prior_starts=139,
        admission='all 24 NEW c0 comparison plus all 32 c1; cached c0 eight not duplicated',
        admission_traces=56, admission_cameras=sum(len(t['actions']) for t in admitted),
        admission_files=[f for trace in admitted for f in trace['files']],
        comparison_summary_path=str(COMPARISON / 'summary.json'), comparison_summary_sha256=sha(COMPARISON / 'summary.json'),
        excluded_starts=sorted(excluded), collector_root=str(COLLECTION), collection_seconds=180,
        planned_new_starts=64, phase_counts=[32, 32], camera_cap=64000, initial_cap=64,
        collector='frozen C1, unchanged epsilon.25 at post-hit boundaries; greedy.8125 others.0625',
        learner='same state, observed options, weights, physical discount, MC initialization, up to32FQI',
        optimization='bincount actual-option presence; one C apply for values/support; cache observed predictions; true independent counts initial/final only',
        optimization_parity_path=str(parity), optimization_parity_sha256=sha(parity),
        fit_seconds=180, fit_cutoff_seconds=168, finalization_reserve_seconds=12,
        all_time_charged=True, final_completed_iteration_only=True, no_hyperparameter_sweep=True,
        no_native_in_learner=True, no_fresh_data=True, source_files=source_files())
    assert sha(PRIOR / 'final-forest.npz') == C1_SHA
    write(output / 'cycle-plan.json', plan)
    print(json.dumps(dict(status='prepared_no_fit', plan_sha256=sha(output / 'cycle-plan.json'),
        admitted_traces=56, admitted_cameras=plan['admission_cameras'], excluded_starts=len(excluded))))


def append(output, approved_plan, approved_collection):
    began = time.monotonic()
    if approved_plan != sha(output / 'cycle-plan.json') or approved_collection != sha(COLLECTION / 'spec.json'):
        raise ValueError('approved fixed learner and collection specs required')
    plan = json.loads((output / 'cycle-plan.json').read_text())
    if (output / 'smdp-data.npz').exists():
        raise FileExistsError('cannot silently append the cycle twice')
    assert all(sha(f['path']) == f['sha256'] for f in plan['source_files'])
    assert sha(PRIOR / 'smdp-data.npz') == plan['prior_data_sha256']
    assert all(sha(f['path']) == f['sha256'] for f in plan['admission_files'])
    # Explicit process-local reader configuration, never a pinned source edit.
    ingestion.COLLECTION = COLLECTION; ingestion.C0_SHA = C1_SHA
    completed = ingestion.validate_completed_collection()
    collected = list(ingestion.collection_traces()); admitted = list(comparison_traces())
    if len(collected) != completed['started'] or sum(len(t['actions']) for t in collected) != completed['transition_cameras']:
        raise ValueError('every started collector2 receipt and camera must be ingested')
    seeds = [t['seed'] for t in collected]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(plan['excluded_starts']):
        raise ValueError('new collection starts must exclude all 142 prior/admitted starts')
    with np.load(PRIOR / 'smdp-data.npz', allow_pickle=False) as saved:
        old = {key:saved[key].copy() for key in saved.files}
    appender = ingestion.Appender(old, prior_seeds())
    for trace in admitted + collected:
        if trace['family'] == 'new_closed_loop_option_exploration':
            trace['family'] = 'new_closed_loop_option_exploration_cycle2'
        appender.add(trace)
    data = appender.finish()
    # Preserve prior observations, labels, start identifiers and timing exactly.
    for key, value in old.items():
        prefix = data[key] if key == 'scales' else data[key][:len(value)]
        assert prefix.tobytes() == value.tobytes(), ('prior array changed', key)
    eligible = data['state_valid'][data['source']]; known = eligible & np.isfinite(data['mc'])
    weights = balanced_weights(data['option'], data['start'], eligible)
    mc_weights = balanced_weights(data['option'], data['start'], known)
    by_option = []
    for option in range(4):
        rows = data['option'] == option; valid = rows & eligible
        by_option.append(dict(option=option, actual_rows=int(rows.sum()), eligible_rows=int(valid.sum()),
            distinct_starts=int(len(np.unique(data['start'][valid]))), known_mc_rows=int((rows & known).sum()),
            unknown_mc_rows=int((valid & ~np.isfinite(data['mc'])).sum()),
            fqi_weight_mass=float(weights[rows].sum()), mc_weight_mass=float(mc_weights[rows].sum())))
    np.savez_compressed(output / 'smdp-data.npz', **data)
    spec = json.loads((PRIOR / 'spec.json').read_text())
    spec.update(data_sha256=sha(output / 'smdp-data.npz'), source_files=plan['source_files'],
        fit_cutoff_seconds=168, finalization_reserve_seconds=12,
        training_optimization=plan['optimization'], optimization_parity_sha256=plan['optimization_parity_sha256'],
        cycle=dict(number=2, plan_sha256=approved_plan, collection_spec_sha256=approved_collection,
            admitted_new_comparisons=56, cached_c0_eight_duplicated=False, prior_data_sha256=plan['prior_data_sha256'],
            collector_model_sha256=C1_SHA, no_initial_policy_replacement=True))
    write(output / 'spec.json', spec)
    write(output / 'data-manifest.json', dict(prior_manifest_path=str(PRIOR / 'data-manifest.json'),
        prior_manifest_sha256=sha(PRIOR / 'data-manifest.json'), added_traces=appender.receipts,
        collection_summary_path=str(COLLECTION / 'summary.json'), collection_summary_sha256=sha(COLLECTION / 'summary.json'),
        start_id_to_seed={value:key for key,value in appender.seed_map.items()}))
    summary = dict(status='appended_no_fit', prior_rows=len(old['source']), combined_rows=len(data['source']),
        eligible_rows=int(eligible.sum()), distinct_starts=len(appender.seed_map), by_option=by_option,
        admitted_traces=56, admitted_cameras=sum(len(t['actions']) for t in admitted),
        collected_traces=len(collected), collected_cameras=completed['transition_cameras'],
        collected_actual_hits=completed['actual_hits'], prior_array_prefixes_byte_identical=True,
        data_bytes=(output / 'smdp-data.npz').stat().st_size, elapsed_seconds=time.monotonic() - began,
        spec_sha256=sha(output / 'spec.json'))
    write(output / 'inventory.json', summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'append', 'fit'])
    parser.add_argument('--output', type=Path, default=DEST)
    parser.add_argument('--approved-plan-sha256'); parser.add_argument('--approved-collection-spec-sha256')
    parser.add_argument('--approved-spec-sha256'); args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args.output)
    elif args.operation == 'append':
        append(args.output, args.approved_plan_sha256, args.approved_collection_spec_sha256)
    else:
        fast_fit(args.output, args.approved_spec_sha256)
