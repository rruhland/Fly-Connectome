"""Same bounded MC+FQI learner with exact cached C-tree queries."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from run_smdp_adaptive_forest import sha, write
from pong_smdp_adaptive_forest import (AdaptiveForest, option_features, balanced_weights,
    discount, sklearn_q_values)
from pong_smdp_fast_training import (training_leaf_cache, fast_q_values,
    independent_counts_from_leaves)


def fit(output, approved):
    spec = json.loads((output / 'spec.json').read_text())
    if approved != sha(output / 'spec.json'):
        raise ValueError('approved final data/spec hash required before sole sequence')
    assert all(sha(s['path']) == s['sha256'] for s in spec['source_files'])
    assert sha(output / 'smdp-data.npz') == spec['data_sha256']
    if (output / 'fit-clock.json').exists():
        raise FileExistsError('the original fit clock cannot restart')
    began = time.monotonic(); deadline = began + 180
    write(output / 'fit-clock.json', dict(started_monotonic=began, deadline_monotonic=deadline,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seconds=180,
        fit_cutoff_seconds=168, finalization_reserve_seconds=12, all_time_charged=True))
    class FitDeadline(Exception):
        pass
    def stop(*_):
        raise FitDeadline()
    prior_handler = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, 168.)
    result = dict(status='started', native_cameras=0, completed_fqi_updates=0,
        initial_mc_completed=False, exact_training_optimization=True,
        final_iteration_selection='last complete before fixed deadline; no outcome selection')
    last = None; initial = None; history = []
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
        leaves, presence, observed_prediction = training_leaf_cache(model, F, options, known)
        initial_counts = independent_counts_from_leaves(model, leaves[known], options[known], starts[known])
        initial = AdaptiveForest.export(model, data['scales'], F[known], options[known], starts[known], initial_counts)
        initial.save(output / 'initial-mc.npz')
        last = model, leaves, presence, 0
        result.update(initial_mc_completed=True, mc_known_rows=int(known.sum()),
            mc_unknown_rows=int((~known).sum()), mc_initial_seconds=time.monotonic() - before)
        next_ids = np.unique(successor[(kind == 0) & (successor >= 0)])
        next_position = np.searchsorted(next_ids, np.maximum(0, successor))
        continuing = (kind == 0) & (successor >= 0)
        prior_cycle = 0.
        for iteration in range(1, 33):
            if time.monotonic() + max(prior_cycle, 2.) + 12 >= deadline:
                result['stop'] = 'insufficient_time_for_next_complete_update'; break
            update_began = time.monotonic()
            qnext, supported = fast_q_values(model, presence, data['states'][next_ids], data['state_valid'][next_ids])
            continuation = np.max(np.where(supported, qnext, 0), axis=1)
            target = np.zeros(len(F))
            target[kind == 1] = beta[kind == 1]
            target[continuing] = beta[continuing] * continuation[next_position[continuing]]
            candidate = ExtraTreesRegressor(**spec['estimator'])
            candidate.fit(F, target, sample_weight=weights)
            candidate_leaves, candidate_presence, candidate_prediction = training_leaf_cache(candidate, F, options)
            if time.monotonic() >= deadline - 12:
                result['stop'] = 'update_not_applied_after_fit_cutoff'; break
            movement = candidate_prediction - observed_prediction
            diagnostic = dict(iteration=iteration, seconds=time.monotonic() - update_began,
                max_q_movement=float(np.max(abs(movement))), rms_q_movement=float(np.sqrt(np.mean(movement ** 2))),
                fitted_target_rmse=float(np.sqrt(np.mean((candidate_prediction - target) ** 2))),
                target_min=float(target.min()), target_max=float(target.max()),
                next_support_fraction_by_option=supported.mean(axis=0).tolist(),
                independent_start_counts_computed=False,
                independent_start_diagnostics='initial/final only; presence is not independent evidence',
                unsupported_next_states=int((~supported.any(axis=1)).sum()),
                rms_q_movement_by_option=[float(np.sqrt(np.mean(movement[options == a] ** 2))) for a in range(4)])
            history.append(diagnostic); prior_cycle = diagnostic['seconds']
            model, leaves, presence = candidate, candidate_leaves, candidate_presence
            observed_prediction = candidate_prediction
            last = model, leaves, presence, iteration
            result['completed_fqi_updates'] = iteration
            write(output / 'iterations.json', history)
        else:
            result['stop'] = 'completed_32_updates'
    except FitDeadline:
        result['stop'] = 'original_fit_cutoff'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
    if last is not None:
        model, leaves, presence, iteration = last
        selected = known if iteration == 0 else np.ones(len(F), bool)
        counts = initial_counts if iteration == 0 else independent_counts_from_leaves(
            model, leaves, options, starts)
        assert np.array_equal(presence, counts > 0), 'fast presence changed actual-option support'
        final = initial if iteration == 0 else AdaptiveForest.export(model, data['scales'],
            F[selected], options[selected], starts[selected], counts)
        final.save(output / 'final-forest.npz')
        # Full portable export parity remains explicit and charged.
        expected = model.predict(F)
        actual, _, _ = final.predict(F, options)
        assert np.array_equal(expected, actual), 'portable exported predictions changed'
        audit_ids = np.linspace(0, len(data['states']) - 1, min(512, len(data['states'])), dtype=int)
        audit_states, audit_valid = data['states'][audit_ids], data['state_valid'][audit_ids]
        reference_q, reference_support, distinct_trees = sklearn_q_values(model, counts, audit_states, audit_valid)
        fast_q, fast_support = fast_q_values(model, presence, audit_states, audit_valid)
        assert np.array_equal(reference_q, fast_q) and np.array_equal(reference_support, fast_support)
        reloaded = AdaptiveForest.load(output / 'final-forest.npz')
        actual_q, actual_support, _ = reloaded.q_values(audit_states, audit_valid)
        assert np.array_equal(reference_q, actual_q) and np.array_equal(reference_support, actual_support)
        result.update(status='fqi_endpoint' if iteration else 'mc_only_endpoint', final_iteration=iteration,
            completed_fqi_updates=iteration,
            training_export_exact_rows=len(F), serialized_all_option_parity_rows=len(audit_states) * 4,
            final_fast_vs_original_q_support_exact=True, final_presence_vs_independence_exact=True,
            final_sha256=sha(output / 'final-forest.npz'), initial_sha256=sha(output / 'initial-mc.npz'),
            final_nodes=int(sum(t.tree_.node_count for t in model.estimators_)),
            diagnostic_support_fraction_by_option=reference_support.mean(0).tolist(),
            final_trees_with_two_starts_mean_by_option=distinct_trees.mean(0).tolist(),
            independent_start_counts_are_diagnostic_only=True)
    else:
        result['status'] = 'no_complete_initial_fit'
    result.update(elapsed_seconds=time.monotonic() - began, within_180_seconds=time.monotonic() <= deadline,
        artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    write(output / 'iterations.json', history); write(output / 'fit-summary.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--approved-spec-sha256', required=True)
    args = parser.parse_args(); fit(args.output, args.approved_spec_sha256)
