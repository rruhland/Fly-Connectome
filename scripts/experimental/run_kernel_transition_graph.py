"""One pinned offline graph fit. No simulator, native observations or tuning."""
import time
ENTRY_TIME = time.monotonic()
import argparse
import hashlib
import json
from pathlib import Path
import sys

# The shared interpreter is installed against another checkout; select this one.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import numpy as np
import scipy.sparse as sp

from pong_kernel_transition_graph import (SCHEMA, SceneMetric, TransitionIndex,
    KernelGraph, elapsed_discount, reconstruct_timing, scheduled_ticks, solve)

SOURCE = Path('/workspace/m1b-experiments/pong-rally-continuous-actor-20261028-r1')
BASELINE = Path('/workspace/m1b-experiments/pong-rally-joint-output-20261028-r1/rbf-model.npz')
BASELINE_SHA = '40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438'
DEFAULT_OUTPUT = Path('/workspace/m1b-experiments/pong-kernel-transition-graph-20261029-r1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_scenes(path):
    with np.load(path, allow_pickle=False) as data:
        return [data['entities'][a:b].copy() for a, b in zip(data['indptr'][:-1], data['indptr'][1:])]


def calibration_metric():
    report = json.loads((SOURCE / 'calibration-report.json').read_text())
    scales = np.maximum(report['0']['standard_deviations'], report['0']['floor'])
    provisional = SceneMetric(scales, 1.)
    distances = []
    calibration_files = []
    for path in sorted((SOURCE / 'calibration').glob('*-scenes.npz')):
        scenes = read_scenes(path)
        vectors, keys = zip(*(provisional.encode(scene, i % 5, False) for i, scene in enumerate(scenes)))
        for i in range(len(scenes) - 5):
            if keys[i] >= 0 and keys[i] == keys[i + 5]:
                distances.append(float(np.linalg.norm(vectors[i] - vectors[i + 5])))
        calibration_files.append(dict(path=str(path), sha256=sha(path), rows=len(scenes)))
    if not distances:
        raise ValueError('no unlabeled five-camera same-mask calibration pairs')
    radius = float(np.quantile(distances, .95))
    return SceneMetric(scales, radius), dict(scales=scales.tolist(), radius=radius,
        radius_quantile=.95, radius_pairs=len(distances), radius_gap_cameras=5,
        files=calibration_files, calibration_report_sha256=sha(SOURCE / 'calibration-report.json'))


def load_training(metric, deadline):
    from pong_fixed_rbf import load_readout
    assert sha(BASELINE) == BASELINE_SHA
    baseline = load_readout(BASELINE)
    vectors, keys, successors, successor_keys = [], [], [], []
    actions, hits, terminals, elapsed, starts, baseline_actions, post_hit = [], [], [], [], [], [], []
    rows_manifest, timing = [], []
    all_scenes, ordinals = [], []
    excluded_finals = verified_terminal_finals = total_rows = total_hits = 0
    source_files = sorted((SOURCE / 'training').glob('*-scores.npz'))
    seeds = sorted({json.loads(p.with_name(p.name.replace('-scores.npz', '.json')).read_text())['seed'] for p in source_files})
    seed_ids = {seed: i for i, seed in enumerate(seeds)}
    for score_path in source_files:
        if time.monotonic() >= deadline:
            raise TimeoutError('data loading exceeded offline fit allocation')
        receipt_path = score_path.with_name(score_path.name.replace('-scores.npz', '.json'))
        scene_path = score_path.with_name(score_path.name.replace('-scores.npz', '-scenes.npz'))
        raw_path = score_path.with_name(score_path.name.replace('-scores.npz', '-raw.npz'))
        receipt = json.loads(receipt_path.read_text())
        scenes = read_scenes(scene_path)
        with np.load(score_path, allow_pickle=False) as scores:
            actual_actions, actual_hits = scores['actions'].copy(), scores['hits'].copy()
        n = len(actual_actions)
        assert len(scenes) == n == receipt['transitions']
        assert np.isin(actual_hits, [0, 1]).all() and int(actual_hits.sum()) == receipt['hits']
        durations, ticks = reconstruct_timing(n, receipt['report']['physics_seconds'])
        last_rally = receipt['report']['rallies'][-1] if receipt['report']['rallies'] else None
        natural_final = bool(receipt['complete'] and last_rally and last_rally['terminated']
            and not last_rally['truncated'] and receipt['report']['episodes'] == 1)
        stop = n if natural_final else n - 1
        excluded_finals += int(not natural_final)
        verified_terminal_finals += int(natural_final)
        if not natural_final:
            assert ticks[-1] == scheduled_ticks(n - 1)
        raw = sp.load_npz(raw_path).tocsr()
        assert raw.shape == (n, 4389)
        earned = False
        for i in range(stop):
            vector, key = metric.encode(scenes[i], i % 5, earned)
            next_earned = earned or bool(actual_hits[i])
            if i + 1 < n:
                successor, successor_key = metric.encode(scenes[i + 1], (i + 1) % 5, next_earned)
            else:
                assert natural_final
                successor, successor_key = np.zeros(16), -1
            lo, hi = raw.indptr[i:i + 2]
            baseline_action, _ = baseline.choose(raw.indices[lo:hi], raw.data[lo:hi])
            vectors.append(vector); keys.append(key)
            successors.append(successor); successor_keys.append(successor_key)
            actions.append(actual_actions[i]); hits.append(actual_hits[i])
            terminals.append(bool(natural_final and i == n - 1))
            elapsed.append(durations[i]); starts.append(seed_ids[receipt['seed']])
            baseline_actions.append(baseline_action); post_hit.append(earned)
            all_scenes.append(scenes[i]); ordinals.append(i)
            earned = next_earned
        total_rows += n; total_hits += int(actual_hits.sum())
        timing.append(dict(tag=receipt['tag'], rows=n, total_ticks=int(ticks.sum()),
            final_ticks=int(ticks[-1]), scheduled_final_ticks=int(scheduled_ticks(n - 1)),
            report_seconds=receipt['report']['physics_seconds'],
            reconstruction_error_seconds=abs(float(durations.sum()) - receipt['report']['physics_seconds'])))
        rows_manifest.append(dict(tag=receipt['tag'], seed=receipt['seed'], eligible_rows=stop,
            verified_natural_final=natural_final, files=[dict(path=str(p), sha256=sha(p))
                for p in (receipt_path, scene_path, score_path, raw_path)]))
    data = dict(vectors=np.asarray(vectors), keys=np.asarray(keys), successors=np.asarray(successors),
        successor_keys=np.asarray(successor_keys), actions=np.asarray(actions), hit=np.asarray(hits, dtype=bool),
        terminal=np.asarray(terminals), elapsed=np.asarray(elapsed), starts=np.asarray(starts),
        baseline_actions=np.asarray(baseline_actions), post_hit=np.asarray(post_hit), ordinals=np.asarray(ordinals))
    return data, all_scenes, dict(files=rows_manifest, timing=timing, source_rows=total_rows,
        source_hits=total_hits, eligible_rows=len(vectors), retained_hits=int(data['hit'].sum()),
        interior_rows=total_rows - len(source_files), verified_terminal_finals=verified_terminal_finals,
        excluded_censored_finals=excluded_finals, distinct_starts=len(seeds),
        unsupported_overflow_source_rows=int((data['keys'] < 0).sum()),
        unsupported_overflow_successor_rows=int(((data['successor_keys'] < 0) & ~data['terminal']).sum()),
        overflow_source_hit_events=int(((data['keys'] < 0) & data['hit']).sum()),
        overflow_source_terminal_events=int(((data['keys'] < 0) & data['terminal']).sum()))


def main():
    global ENTRY_TIME
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume-initialization-failure', action='store_true')
    parser.add_argument('--recovery-of', type=Path)
    args = parser.parse_args()
    output = args.output
    if output.exists() and args.resume_initialization_failure:
        summary_path = output / 'summary.json'
        prior = json.loads(summary_path.read_text())
        if (prior.get('status') != 'failed' or prior.get('error') !=
                "ModuleNotFoundError: No module named 'fly_connectome.control'"
                or (output / 'initialization-failure.json').exists()):
            raise ValueError('only the recorded pre-data import failure can resume once')
        # Preserve the original allocation, conservatively subtracting 10ms for
        # the interval between computing elapsed time and writing the receipt.
        original_epoch = summary_path.stat().st_mtime - prior['elapsed_seconds'] - .01
        ENTRY_TIME = time.monotonic() - (time.time() - original_epoch)
        summary_path.rename(output / 'initialization-failure.json')
        (output / 'spec.json').rename(output / 'initialization-failure-spec.json')
    elif output.exists():
        raise FileExistsError('single-fit artifact directory already exists; no automatic restart')
    else:
        output.mkdir(parents=True)
    deadline = ENTRY_TIME + 60.
    spec = dict(schema=SCHEMA, input_root=str(SOURCE / 'training'), baseline=str(BASELINE),
        baseline_sha256=BASELINE_SHA, all_eligible_rows=True, reward_oversampling=False,
        neighbors_per_action=8, independent_start_minimum=2, exact_strata=['entity_presence',
            'observed_masks', 'velocity_known_masks', 'past_own_hit', 'camera_ordinal_mod_5'],
        coordinate_order='horizontal coordinate1 primary, vertical coordinate0 secondary',
        radius_rule='p95 Euclidean distance of same-mask unlabeled calibration scenes 5 cameras apart',
        weights='normalized exp(-0.5*(distance/radius)^2) within fixed radius; zero otherwise',
        half_life_seconds=3.2, discount='2**(-actual_elapsed_seconds/3.2)',
        hits_absorb_at_one=True, verified_natural_ends_absorb_at_zero=True,
        unsupported_boundary_value=0, unsupported_boundaries_are_natural_misses=False,
        exclude_censored_final_without_successor=True, baseline_scores_in_features=False,
        overflow='retain row, source/query unsupported, successor unknown model boundary; capacity remains4',
        runtime='baseline until past actual own hit; base and selected action both supported; strict value gain',
        fit_seconds=60, max_iterations=4096, tolerance=1e-8,
        negligible_support='zero post-hit source states with supported baseline and any supported alternative',
        no_native=True, no_backprop=True, no_sweep=True,
        sources=[dict(path=str(Path(__file__)), sha256=sha(__file__)),
            dict(path=str(Path(__file__).with_name('pong_kernel_transition_graph.py')),
                sha256=sha(Path(__file__).with_name('pong_kernel_transition_graph.py')))])
    if args.recovery_of is not None:
        prior_root = args.recovery_of.resolve()
        prior_summary = json.loads((prior_root / 'summary.json').read_text())
        if prior_summary.get('status') != 'failed' or prior_summary.get('native_cameras') != 0:
            raise ValueError('explicit recovery requires a failed pre-fit offline run')
        assert not (prior_root / 'graph.npz').exists(), 'recovery cannot replace a prior learned fit'
        spec['recovery'] = dict(original_root=str(prior_root), separate_allocation_seconds=60,
            actual_fit_index=1, authorization='Parent explicitly authorized separate60s after two pre-fit initialization failures',
            changed_sources='Runner selects checkout src; overflow states retained but unsupported at fixed capacity4',
            prior_files=[dict(path=str(prior_root / name), sha256=sha(prior_root / name)) for name in
                ('initialization-failure.json', 'initialization-failure-spec.json', 'summary.json', 'spec.json')])
    write_json(output / 'spec.json', spec)
    result = dict(status='started', native_cameras=0, fit_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    try:
        metric, calibration = calibration_metric()
        write_json(output / 'calibration.json', calibration)
        data, scenes, manifest = load_training(metric, deadline - 3)
        write_json(output / 'data-manifest.json', manifest)
        index = TransitionIndex(data['vectors'], data['keys'], data['actions'], data['starts'], metric.radius)
        source_neighbors = index.query(data['vectors'], data['keys'], deadline - 3)
        successor_neighbors = index.query(data['successors'], data['successor_keys'], deadline - 3)
        support = source_neighbors.supported
        baseline_supported = support[np.arange(len(support)), data['baseline_actions']]
        alternatives = support.copy()
        alternatives[np.arange(len(support)), data['baseline_actions']] = False
        usable = data['post_hit'] & baseline_supported & alternatives.any(axis=1)
        continuation = ~data['hit'] & ~data['terminal']
        coverage = dict(source_supported_action_cells=int(support.sum()), source_action_cells=int(support.size),
            source_support_by_action=support.sum(axis=0).tolist(), post_hit_rows=int(data['post_hit'].sum()),
            post_hit_base_and_alternative_supported=int(usable.sum()),
            post_hit_base_and_alternative_fraction=float(usable.sum() / max(1, data['post_hit'].sum())),
            successor_no_action_supported=int((continuation & ~successor_neighbors.supported.any(axis=1)).sum()),
            continuation_rows=int(continuation.sum()),
            unsupported_overflow_source_rows=int((data['keys'] < 0).sum()),
            unsupported_overflow_successor_rows=int((continuation & (data['successor_keys'] < 0)).sum()),
            distinct_start_support_is_not_counterfactual_guarantee=True)
        write_json(output / 'coverage.json', coverage)
        result.update(coverage=coverage, eligible_rows=len(support), retained_hits=int(data['hit'].sum()),
            natural_terminal_rows=int(data['terminal'].sum()), radius=metric.radius,
            overflow_source_hit_events=manifest['overflow_source_hit_events'],
            overflow_source_terminal_events=manifest['overflow_source_terminal_events'])
        if not usable.any():
            result.update(status='negligible_support', fit=None)
        else:
            values, fitted = solve(successor_neighbors, data['hit'], data['terminal'],
                elapsed_discount(data['elapsed']), deadline - 3)
            result['fit'] = fitted
            graph = KernelGraph(metric, index, values)
            graph.save(output / 'graph.npz')
            source_q = source_neighbors.q_values(values)
            candidate = np.argmax(np.where(support, source_q, -np.inf), axis=1)
            candidate_q = source_q[np.arange(len(support)), candidate]
            baseline_q = source_q[np.arange(len(support)), data['baseline_actions']]
            changes = usable & (candidate_q > baseline_q) & (candidate != data['baseline_actions'])
            np.savez_compressed(output / 'source-diagnostics.npz', baseline_actions=data['baseline_actions'],
                q=source_q, supported=support, changed=changes, post_hit=data['post_hit'],
                actions=data['actions'], hit=data['hit'], terminal=data['terminal'], elapsed=data['elapsed'])
            reloaded = KernelGraph.load(output / 'graph.npz')
            parity_rows = np.linspace(0, len(scenes) - 1, min(64, len(scenes)), dtype=int)
            for i in parity_rows:
                before = graph.inspect(scenes[i], int(data['ordinals'][i] % 5), bool(data['post_hit'][i]), int(data['baseline_actions'][i]))
                after = reloaded.inspect(scenes[i], int(data['ordinals'][i] % 5), bool(data['post_hit'][i]), int(data['baseline_actions'][i]))
                assert before == after, 'serialized graph query mismatch'
                assert np.allclose(before['q'], source_q[i], atol=1e-14, rtol=0)
            result.update(status='converged' if fitted['converged'] else 'fit_failed_budget_or_iteration',
                changed_saved_source_decisions=int(changes.sum()), serialization_query_parity_rows=len(parity_rows),
                graph_sha256=sha(output / 'graph.npz'), value_min=float(values.min()), value_max=float(values.max()))
    except TimeoutError as error:
        result.update(status='offline_budget_exceeded', error=str(error))
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - ENTRY_TIME
        result['within_60_seconds'] = result['elapsed_seconds'] <= 60
        result['new_artifact_bytes'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        write_json(output / 'summary.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
