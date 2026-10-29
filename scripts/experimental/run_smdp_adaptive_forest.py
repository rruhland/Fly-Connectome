"""Inventory first; one explicitly approved, bounded SMDP fitted-Q sequence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import scipy.sparse as sp
from pong_smdp_adaptive_forest import (SCHEMA, STATE_DIM, state_features,
    option_features, observed_windows, observed_mc_returns, balanced_weights, discount)
from pong_kernel_transition_graph import reconstruct_timing

ROOT = Path('/workspace/m1b-experiments')
CONTINUOUS = ROOT / 'pong-rally-continuous-actor-20261028-r1'
PAIRED = ROOT / 'pong-paired-option-forest-20261029-r1'
BASELINE = ROOT / 'pong-rally-joint-output-20261028-r1/rbf-model.npz'
EXPECTED = '40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438'
DEST = ROOT / 'pong-smdp-adaptive-forest-20261029-r1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def scenes(path):
    with np.load(path, allow_pickle=False) as data:
        return [data['entities'][a:b].copy() for a, b in zip(data['indptr'][:-1], data['indptr'][1:])]


def old_traces():
    from pong_fixed_rbf import load_readout
    assert sha(BASELINE) == EXPECTED
    baseline = load_readout(BASELINE)
    for path in sorted((CONTINUOUS / 'training').glob('*-scores.npz')):
        prefix = path.name.removesuffix('-scores.npz')
        receipt_path = path.with_name(prefix + '.json')
        scene_path = path.with_name(prefix + '-scenes.npz')
        raw_path = path.with_name(prefix + '-raw.npz')
        receipt = json.loads(receipt_path.read_text())
        with np.load(path, allow_pickle=False) as data:
            actions, hits = data['actions'].copy(), data['hits'].copy()
        n = len(actions)
        visual = scenes(scene_path)
        assert len(visual) == n == receipt['transitions']
        elapsed, _ = reconstruct_timing(n, receipt['report']['physics_seconds'])
        natural = np.zeros(n, bool)
        if receipt['complete']:
            rally = receipt['report']['rallies'][-1]
            natural[-1] = rally['terminated'] and not rally['truncated']
        raw = sp.load_npz(raw_path).tocsr()
        base_actions = np.empty(n, np.int8)
        for i in range(n):
            lo, hi = raw.indptr[i:i + 2]
            base_actions[i] = baseline.choose(raw.indices[lo:hi], raw.data[lo:hi])[0]
        yield dict(name=prefix, family='continuous', seed=receipt['seed'], ordinal=0,
            past_hit=False, scenes=visual, actions=actions, hits=hits, natural=natural,
            elapsed=elapsed, baseline=base_actions,
            files=[dict(path=str(p), sha256=sha(p)) for p in (path, receipt_path, scene_path, raw_path)])


def paired_traces():
    for site_path in sorted((PAIRED / 'acquisition').glob('site_??.json')):
        site = json.loads(site_path.read_text())['site']
        assert site['previous_hit'] < site['anchor']
        for option in range(4):
            stem = site_path.with_name(site_path.stem + f'-option_{option}')
            data_path = Path(str(stem) + '-transitions.npz')
            scene_path = Path(str(stem) + '-scenes.npz')
            if not data_path.exists():
                continue
            with np.load(data_path, allow_pickle=False) as data:
                actions, hits, natural = data['action'].copy(), data['hits'].copy(), data['terminated'].copy()
                elapsed, ticks = data['elapsed_seconds'].copy(), data['physics_ticks'].copy()
                base_actions = np.argmax(data['baseline_q'], axis=1).astype(np.int8)
            visual = scenes(scene_path)
            assert len(visual) == len(actions) + 1 == len(base_actions)
            assert np.allclose(elapsed, ticks / 120, atol=1e-15, rtol=0)
            yield dict(name=stem.name, family='paired', seed=site['seed'], ordinal=site['anchor'],
                past_hit=True, scenes=visual, actions=actions, hits=hits, natural=natural,
                elapsed=elapsed, baseline=base_actions[:-1],
                files=[dict(path=str(p), sha256=sha(p)) for p in (site_path, data_path, scene_path)])


def inventory(output):
    began = time.monotonic()
    if output.exists():
        raise FileExistsError('inventory destination already exists')
    output.mkdir(parents=True)
    report = json.loads((CONTINUOUS / 'calibration-report.json').read_text())
    scales = np.maximum(report['0']['standard_deviations'], report['0']['floor'])
    states, valid_states, state_start = [], [], []
    sources, successors, options, durations, kinds, start_ids, lengths, mc = [], [], [], [], [], [], [], []
    rows = []; start_map = {}; rejected = {}
    for trace_index, trace in enumerate((*old_traces(), *paired_traces())):
        seed = int(trace['seed']); start_id = start_map.setdefault(seed, len(start_map))
        n = len(trace['actions'])
        assert np.isin(trace['hits'], [0, 1]).all() and n > 0
        prior = np.r_[bool(trace['past_hit']), np.logical_or.accumulate(trace['hits'].astype(bool)) | bool(trace['past_hit'])]
        offset = len(states)
        for i, scene in enumerate(trace['scenes']):
            feature = state_features(scene, (trace['ordinal'] + i) % 5, prior[i], scales)
            states.append(np.zeros(STATE_DIM, np.float32) if feature is None else feature)
            valid_states.append(feature is not None); state_start.append(start_id)
        successor_available = np.arange(n) + 1 < len(trace['scenes'])
        windows, failures = observed_windows(trace['actions'], trace['hits'], trace['natural'],
            trace['baseline'], successor_available)
        for name, count in failures.items():
            rejected[name] = rejected.get(name, 0) + count
        mc_return = observed_mc_returns(trace['hits'], trace['natural'], trace['elapsed'])
        cumulative = np.r_[0., np.cumsum(trace['elapsed'])]
        local_counts = np.zeros(4, int)
        for source, option, end, kind in windows:
            sources.append(offset + source)
            successors.append(offset + end + 1 if end + 1 < len(trace['scenes']) else -1)
            options.append(option); kinds.append(kind); start_ids.append(start_id)
            durations.append(cumulative[end + 1] - cumulative[source]); lengths.append(end - source + 1)
            mc.append(mc_return[source]); local_counts[option] += 1
        rows.append(dict(name=trace['name'], family=trace['family'], seed=seed, start_id=start_id,
            source_cameras=n, saved_scenes=len(trace['scenes']), own_hit_events=int(trace['hits'].sum()),
            natural_end_events=int(trace['natural'].sum()), option_counts=local_counts.tolist(),
            rejected=failures, files=trace['files']))
    states = np.asarray(states, np.float32); valid_states = np.asarray(valid_states, bool)
    source = np.asarray(sources, np.int32); successor = np.asarray(successors, np.int32)
    options = np.asarray(options, np.int8); start_ids = np.asarray(start_ids, np.int32)
    kinds = np.asarray(kinds, np.int8); durations = np.asarray(durations); lengths = np.asarray(lengths, np.int8)
    mc = np.asarray(mc); usable = valid_states[source]
    weights = balanced_weights(options, start_ids, usable)
    mc_usable = usable & np.isfinite(mc)
    mc_weights = balanced_weights(options, start_ids, mc_usable)
    next_valid = (successor >= 0) & valid_states[np.maximum(0, successor)]
    by_option = []
    for option in range(4):
        selected = options == option; eligible = selected & usable
        by_option.append(dict(option=option, rows=int(selected.sum()), eligible_rows=int(eligible.sum()),
            distinct_starts=int(len(np.unique(start_ids[selected]))),
            eligible_distinct_starts=int(len(np.unique(start_ids[eligible]))),
            source_overflow_or_empty=int((selected & ~usable).sum()),
            first_hit_windows=int((selected & (kinds == 1)).sum()),
            natural_end_windows=int((selected & (kinds == 2)).sum()),
            full_nonterminal_windows=int((selected & (kinds == 0)).sum()),
            unknown_successor_boundaries=int((eligible & (kinds == 0) & ~next_valid).sum()),
            known_mc_rows=int((selected & mc_usable).sum()), unknown_mc_rows=int((eligible & ~np.isfinite(mc)).sum()),
            raw_row_fraction=float(selected.mean()), proposed_fqi_weight_mass=float(weights[selected].sum()),
            proposed_mc_weight_mass=float(mc_weights[selected].sum()),
            weight_min=float(weights[eligible].min()) if eligible.any() else None,
            weight_max=float(weights[eligible].max()) if eligible.any() else None))
    source_options = np.zeros((len(states), 4), bool)
    source_options[source, options] = True
    represented_sources = source_options.any(axis=1)
    coverage = {str(i):int((represented_sources & (source_options.sum(1) == i)).sum()) for i in range(1, 5)}
    dataset = output / 'smdp-data.npz'
    np.savez_compressed(dataset, states=states, state_valid=valid_states, state_start=np.asarray(state_start, np.int32),
        source=source, successor=successor, option=options, elapsed=durations, kind=kinds,
        start=start_ids, cameras=lengths, mc=mc, scales=scales)
    spec = dict(schema=SCHEMA, baseline_path=str(BASELINE), baseline_sha256=EXPECTED,
        data_sha256=sha(dataset), state_dim=STATE_DIM, capacity=4, empty_or_overflow='unsupported baseline fallback',
        state='horizontal sorted anonymous sixfield rows, graph-scaled four continuous fields, presence, past own hit, public phase onehot5',
        option_encoding='onehot4; options0/1/2 constantcommand; option3 frozen40069 onecamera',
        macro='first ownhit OR naturalend OR16cameras; no changing-action or shortcensored relabel',
        physical_target='E[2**(-elapsed time through next observed own hit /3.2)]; naturalend0',
        initialization='one actual BEHAVIOR-tail MC return forest fit; unknown tails omitted, no graphdistillation',
        update='up to32 genuine FQI refits; hit target=discount(fullactualelapsed); naturalend0; otherwisediscount*maxsupportedQnext',
        estimator=dict(n_estimators=32, min_samples_leaf=4, max_depth=None, max_features=1.0,
            bootstrap=False, criterion='squared_error', random_state=2026102903, n_jobs=1),
        sample_weights='equal mass per represented option, equal per distinct start within option, uniform within start-option',
        support='actual queried-option row present in strict majority17of32 queried leaves; independent-start counts diagnostic only',
        runtime='baseline until actualownhit; baseoption3 and selectedoption both supported and strictgain; commandholds stop early on ownhit/naturalend',
        fit_seconds=180, final_completed_iteration_only=True, save_initial_and_final_only=True,
        at_least_one_fqi_for_smdp_claim=True, no_native=True, no_backprop=True, no_sweep=True,
        source_files=[dict(path=str(Path(__file__)), sha256=sha(__file__)),
            dict(path=str(Path(__file__).with_name('pong_smdp_adaptive_forest.py')),
                sha256=sha(Path(__file__).with_name('pong_smdp_adaptive_forest.py')))])
    write(output / 'spec.json', spec)
    summary = dict(status='inventory_only_no_fit', traces=len(rows), source_cameras=sum(r['source_cameras'] for r in rows),
        saved_states=len(states), distinct_starts=len(start_map), smdp_rows=len(source),
        usable_rows=int(usable.sum()), by_option=by_option, source_option_count_histogram=coverage,
        source_overflow_or_empty_hits=int((~usable & (kinds == 1)).sum()),
        source_overflow_or_empty_natural_ends=int((~usable & (kinds == 2)).sum()),
        rejected=rejected, data_bytes=dataset.stat().st_size, elapsed_seconds=time.monotonic() - began,
        spec_sha256=sha(output / 'spec.json'))
    write(output / 'data-manifest.json', dict(traces=rows, start_id_to_seed={v:k for k,v in start_map.items()}))
    write(output / 'inventory.json', summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def fit(output, approved):
    import signal
    from sklearn.ensemble import ExtraTreesRegressor
    from pong_smdp_adaptive_forest import AdaptiveForest, leaf_independence, sklearn_q_values
    spec = json.loads((output / 'spec.json').read_text())
    if approved != sha(output / 'spec.json'):
        raise ValueError('parent must approve the final pinned inventory/spec before fitting')
    assert all(sha(s['path']) == s['sha256'] for s in spec['source_files'])
    assert sha(output / 'smdp-data.npz') == spec['data_sha256']
    clock = output / 'fit-clock.json'
    if clock.exists():
        raise FileExistsError('one sequence only; original fit clock cannot restart')
    began = time.monotonic(); deadline = began + 180
    write(clock, dict(started_monotonic=began, deadline_monotonic=deadline,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seconds=180))
    class FitDeadline(Exception):
        pass
    def stop(*_):
        raise FitDeadline()
    previous_handler = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, 172.)
    result = dict(status='started', native_cameras=0, completed_fqi_updates=0,
        initial_mc_completed=False, final_iteration_selection='last complete before fixed deadline; no outcome selection')
    history = []; last = None; initial = None
    try:
        with np.load(output / 'smdp-data.npz', allow_pickle=False) as saved:
            data = {key:saved[key].copy() for key in saved.files}
        eligible = data['state_valid'][data['source']]
        source = data['source'][eligible]; successor = data['successor'][eligible]
        options = data['option'][eligible]; starts = data['start'][eligible]
        kind = data['kind'][eligible]; beta = discount(data['elapsed'][eligible])
        mc = data['mc'][eligible]; known = np.isfinite(mc)
        F = option_features(data['states'][source], options)
        weights = balanced_weights(options, starts)
        mc_weights = balanced_weights(options, starts, known)
        before = time.monotonic()
        model = ExtraTreesRegressor(**spec['estimator'])
        model.fit(F[known], mc[known], sample_weight=mc_weights[known])
        counts = leaf_independence(model, F[known], options[known], starts[known])
        initial = AdaptiveForest.export(model, data['scales'], F[known], options[known], starts[known], counts)
        initial.save(output / 'initial-mc.npz')
        last = (model, counts, 0)
        result.update(initial_mc_completed=True, mc_known_rows=int(known.sum()),
            mc_unknown_rows=int((~known).sum()), mc_initial_seconds=time.monotonic() - before)
        next_ids = np.unique(successor[(kind == 0) & (successor >= 0)])
        next_position = np.searchsorted(next_ids, np.maximum(0, successor))
        continuing = (kind == 0) & (successor >= 0)
        previous_cycle = 0.
        for iteration in range(1, 33):
            if time.monotonic() + max(previous_cycle, 2.) + 8 >= deadline:
                result['stop'] = 'insufficient_time_for_next_complete_update'; break
            update_began = time.monotonic()
            qnext, supported, two_starts = sklearn_q_values(model, counts,
                data['states'][next_ids], data['state_valid'][next_ids])
            continuation = np.max(np.where(supported, qnext, 0), axis=1)
            target = np.zeros(len(F))
            target[kind == 1] = beta[kind == 1]
            target[continuing] = beta[continuing] * continuation[next_position[continuing]]
            old_prediction = model.predict(F)
            candidate = ExtraTreesRegressor(**spec['estimator'])
            candidate.fit(F, target, sample_weight=weights)
            candidate_counts = leaf_independence(candidate, F, options, starts)
            prediction = candidate.predict(F)
            if time.monotonic() >= deadline - 8:
                result['stop'] = 'update_not_applied_after_fit_cutoff'; break
            movement = prediction - old_prediction
            diagnostic = dict(iteration=iteration, seconds=time.monotonic() - update_began,
                max_q_movement=float(np.max(abs(movement))), rms_q_movement=float(np.sqrt(np.mean(movement ** 2))),
                fitted_target_rmse=float(np.sqrt(np.mean((prediction - target) ** 2))),
                target_min=float(target.min()), target_max=float(target.max()),
                next_support_fraction_by_option=supported.mean(axis=0).tolist(),
                next_trees_with_two_starts_mean_by_option=two_starts.mean(axis=0).tolist(),
                unsupported_next_states=int((~supported.any(axis=1)).sum()),
                rms_q_movement_by_option=[float(np.sqrt(np.mean(movement[options == a] ** 2))) for a in range(4)])
            history.append(diagnostic)
            previous_cycle = diagnostic['seconds']
            model, counts = candidate, candidate_counts
            last = model, counts, iteration
            result['completed_fqi_updates'] = iteration
            write(output / 'iterations.json', history)
        else:
            result['stop'] = 'completed_32_updates'
    except FitDeadline:
        result['stop'] = 'original_fit_cutoff'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    if last is not None:
        model, counts, iteration = last
        chosen = known if iteration == 0 else np.ones(len(F), bool)
        final = initial if iteration == 0 else AdaptiveForest.export(model, data['scales'],
            F[chosen], options[chosen], starts[chosen], counts)
        final.save(output / 'final-forest.npz')
        expected = model.predict(F)
        actual, _, _ = final.predict(F, options)
        assert np.array_equal(expected, actual), 'portable forest differs from sklearn on training queries'
        reloaded = AdaptiveForest.load(output / 'final-forest.npz')
        audit_ids = np.linspace(0, len(data['states']) - 1, min(512, len(data['states'])), dtype=int)
        audit_states = data['states'][audit_ids]
        reference, support, counts2 = sklearn_q_values(model, counts, audit_states, data['state_valid'][audit_ids])
        actual_q, actual_support, _ = reloaded.q_values(audit_states, data['state_valid'][audit_ids])
        assert np.array_equal(reference, actual_q) and np.array_equal(support, actual_support)
        result.update(status='fqi_endpoint' if iteration else 'mc_only_endpoint', final_iteration=iteration,
            training_export_exact_rows=len(F), serialized_all_option_parity_rows=len(audit_states) * 4,
            final_sha256=sha(output / 'final-forest.npz'), initial_sha256=sha(output / 'initial-mc.npz'),
            final_nodes=int(sum(t.tree_.node_count for t in model.estimators_)),
            diagnostic_support_fraction_by_option=support.mean(0).tolist(),
            independent_start_counts_are_diagnostic_only=True)
    else:
        result['status'] = 'no_complete_initial_fit'
    result.update(elapsed_seconds=time.monotonic() - began,
        within_180_seconds=time.monotonic() <= deadline,
        artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    write(output / 'iterations.json', history)
    write(output / 'fit-summary.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['inventory', 'fit'])
    parser.add_argument('--output', type=Path, default=DEST)
    parser.add_argument('--approved-spec-sha256')
    args = parser.parse_args()
    if args.operation == 'inventory':
        inventory(args.output)
    else:
        fit(args.output, args.approved_spec_sha256)


if __name__ == '__main__':
    main()
