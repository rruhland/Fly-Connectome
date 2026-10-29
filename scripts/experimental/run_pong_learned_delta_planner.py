"""One pinned fit/export/smoke for the learned visual delta planner."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.linalg import solve
from sklearn.ensemble import ExtraTreesRegressor
from run_smdp_adaptive_forest import sha, write
from pong_learned_delta_planner import (DeltaPlannerModel, CompiledForest, inputs,
    scheduled_seconds, TAIL_SHA, SCHEMA)

ROOT = Path('/workspace/m1b-experiments')
DATA = ROOT / 'pong-delta-dynamics-data-20261029-r1'
DEST = ROOT / 'pong-learned-delta-planner-20261029-r1'
TAIL = ROOT / 'pong-smdp-closed-loop-refit-20261029-r1/final-forest.npz'
DATA_SHA = '69a126e0385f75c45617813cacf859e53cdf7108e08e17e20a2536b0f8d9fd20'


def group_weights(start):
    unique, inverse, counts = np.unique(start, return_inverse=True, return_counts=True)
    return 1 / (len(unique) * counts[inverse])


def positive_floor(values):
    positive = values[values > 0]
    if not len(positive):
        raise ValueError('no variable unlabeled predictive coordinates')
    floor = .1 * float(np.median(positive))
    return np.maximum(values, floor), floor


def masks(data):
    public = scheduled_seconds(data['ordinal'])
    if not np.array_equal(public, data['planned_dt']):
        raise ValueError('public phase and staged planned interval disagree')
    if not np.array_equal(np.argmax(data['state'][data['tail_valid'], 29:], axis=1),
            data['ordinal'][data['tail_valid']] % 5):
        raise ValueError('visual feature phase is not aligned with source public ordinal')
    full = np.isclose(data['actualdt'], public, rtol=0, atol=1e-14)
    if not np.array_equal(full, data['full_interval']):
        raise ValueError('full interval flags disagree with actual timing')
    hit = data['hits'].astype(bool)
    end = data['natural'].astype(bool) & ~hit
    event = data['tail_valid'] & (full | hit | end)
    world = data['tail_valid'] & data['has_next'] & data['next_geometry_valid'] & full & ~hit & ~end
    return world, event, hit, end


def prepare(output):
    if output.exists():
        raise FileExistsError('one fixed learned-delta pilot only')
    output.mkdir(parents=True)
    assert sha(DATA / 'transitions.npz') == DATA_SHA and sha(TAIL) == TAIL_SHA
    inventory = json.loads((DATA / 'inventory.json').read_text())
    assert inventory['source_frozen'] is True and inventory['data_sha256'] == DATA_SHA
    source = [Path(__file__), Path(__file__).with_name('pong_learned_delta_planner.py'),
        Path(__file__).with_name('pong_smdp_adaptive_forest.py'), Path(__file__).with_name('pong_smdp_fast_training.py')]
    spec = dict(schema=SCHEMA, data_path=str(DATA / 'transitions.npz'), data_sha256=DATA_SHA,
        data_manifest_sha256=sha(DATA / 'manifest.json'), data_inventory_sha256=sha(DATA / 'inventory.json'),
        tail_path=str(TAIL), tail_sha256=TAIL_SHA,
        learned_tail='frozen C1 option3 approximate Q, trained with future max; not guaranteed baseline value',
        inputs='scene34 + currentcommandonehot3 + previous16actualcommandonehots48 + publicplanneddt + dt*scene28 + dt*command3 =117',
        event_extra='all6 generic unordered pair4coordinate differences plus pairpresence =30;147total',
        elapsed_input='public scheduled interval only; actual clipped duration is target metadata only',
        world_targets='all28 sorted whole-scene deltas; no identity correspondence; no hit/end and complete sensor interval only',
        world_masks='retain presence/cardinality and observed/known changes; empty successor permitted; empty/overflow sources unsupported',
        ridge_regularization='1e-3*mean slope normal-matrix diagonal; unpenalized intercept',
        scaling='weighted input standard deviations and delta RMS, each floor0.1medianpositive; no reward-dependent scaling',
        world_estimator=dict(n_estimators=8, max_depth=12, min_samples_leaf=32, max_features=1/3,
            bootstrap=False, criterion='squared_error', random_state=2026102905, n_jobs=1),
        event_estimator=dict(n_estimators=16, max_depth=16, min_samples_leaf=8, max_features=1.,
            bootstrap=False, criterion='squared_error', random_state=2026102906, n_jobs=1),
        event_targets=['ownhit', 'naturalend_without_ownhit', 'ownhit*beta_actualdt', 'noevent*beta_actualdt'],
        beta='2**(-actualdt/3.2)', event_weights='equal start mass then uniform rows within start; NO class balancing or positive oversampling',
        world_weights='equal start mass then uniform rows within start',
        competing_mass='Gnext=G+discounted_survival_so_far*discounted_hit_mass; remaining*=discounted_noevent_mass',
        binary_decode='>=.5 for presence/observed/known; absent rows zero; unknown velocity zero; horizontal content sort',
        planner=dict(horizon=8, beam=8, primitive_actions=3, minimum_per_initial_action=1,
            pruning='event mass plus C1 option3 tail at every depth', execution='first command then actual observation'),
        fallback='baseline before first actual hit; empty/overflow current scene; unsupported current tail; invalid predictions/unsupported tail branches rejected; absent baseline-root reference',
        no_geometry_cost=True, no_object_roles=True, no_simulator_lookahead=True, no_expert_labels=True,
        fit_seconds=180, fitting_cutoff_seconds=160, finalization_reserve_seconds=20, all_time_charged=True,
        no_sweep=True, no_native=True, source_files=[dict(path=str(p), sha256=sha(p)) for p in source])
    write(output / 'spec.json', spec)
    print(json.dumps(dict(status='prepared_no_fit', spec_sha256=sha(output / 'spec.json'))))


def fit(output, approved):
    spec = json.loads((output / 'spec.json').read_text())
    if approved != sha(output / 'spec.json'):
        raise ValueError('parent-approved fixed spec required')
    assert all(sha(s['path']) == s['sha256'] for s in spec['source_files'])
    assert sha(spec['data_path']) == spec['data_sha256'] and sha(spec['tail_path']) == TAIL_SHA
    assert sha(DATA / 'manifest.json') == spec['data_manifest_sha256']
    assert sha(DATA / 'inventory.json') == spec['data_inventory_sha256']
    assert json.loads((DATA / 'inventory.json').read_text())['source_frozen'] is True
    if (output / 'fit-clock.json').exists():
        raise FileExistsError('no automatic fit restart')
    began = time.monotonic(); deadline = began + 180
    write(output / 'fit-clock.json', dict(started_monotonic=began, deadline_monotonic=deadline,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seconds=180))
    class Deadline(Exception):
        pass
    def stop(*_):
        raise Deadline()
    previous_handler = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, 160)
    result = dict(status='started', native_cameras=0)
    try:
        with np.load(spec['data_path'], allow_pickle=False) as saved:
            data = {key:saved[key].copy() for key in saved.files}
        world, event, hit, end = masks(data)
        iw = np.flatnonzero(world); ie = np.flatnonzero(event)
        X, _ = inputs(data['state'][iw], data['history16'][iw], data['action'][iw])
        weights = group_weights(data['start'][iw])
        mean = weights @ X
        scale, input_floor = positive_floor(np.sqrt(weights @ ((X - mean) ** 2)))
        Z = (X - mean) / scale
        delta = data['next_state'][iw, :28].astype(np.float64) - data['state'][iw, :28]
        delta_scale, delta_floor = positive_floor(np.sqrt(weights @ (delta ** 2)))
        Y = delta / delta_scale
        design = np.column_stack((Z, np.ones(len(Z))))
        normal = design.T @ (weights[:, None] * design)
        regularization = .001 * float(np.trace(normal[:-1, :-1]) / (len(normal) - 1))
        normal[np.arange(len(normal) - 1), np.arange(len(normal) - 1)] += regularization
        ridge = solve(normal, design.T @ (weights[:, None] * Y), assume_a='pos', check_finite=False)
        residual_targets = Y - design @ ridge
        delta_model = ExtraTreesRegressor(**spec['world_estimator'])
        delta_began = time.monotonic(); delta_model.fit(Z, residual_targets, sample_weight=weights)
        result['delta_tree_fit_seconds'] = time.monotonic() - delta_began
        _, E = inputs(data['state'][ie], data['history16'][ie], data['action'][ie])
        beta = np.exp2(-data['actualdt'][ie] / 3.2)
        h, e = hit[ie], end[ie]
        event_targets = np.column_stack((h, e, h * beta, (~h & ~e) * beta))
        event_model = ExtraTreesRegressor(**spec['event_estimator'])
        event_began = time.monotonic(); event_model.fit(E, event_targets, sample_weight=group_weights(data['start'][ie]))
        result['event_tree_fit_seconds'] = time.monotonic() - event_began
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Finalization is charged, and no new training begins here.
        residual = CompiledForest.export(delta_model); events = CompiledForest.export(event_model)
        audit_world = np.linspace(0, len(iw) - 1, min(2048, len(iw)), dtype=int)
        audit_event = np.linspace(0, len(ie) - 1, min(2048, len(ie)), dtype=int)
        assert np.array_equal(residual.predict(Z[audit_world]), delta_model.predict(Z[audit_world]))
        assert np.array_equal(events.predict(E[audit_event]), event_model.predict(E[audit_event]))
        model = DeltaPlannerModel(data['scales'], mean, scale, delta_scale, ridge, residual, events, spec['tail_path'])
        path = output / 'planner-model.npz'; model.save(path)
        restored = DeltaPlannerModel.load(path)
        probe_ids = iw[data['state'][iw, 28] > 0]
        _, first = np.unique(data['start'][probe_ids], return_index=True)
        probe_ids = probe_ids[first]
        probe_ids = probe_ids[np.linspace(0, len(probe_ids) - 1, min(32, len(probe_ids)), dtype=int)]
        before = [model.plan(data['state'][i], data['history16'][i], int(data['action'][i])) for i in probe_ids]
        after = [restored.plan(data['state'][i], data['history16'][i], int(data['action'][i])) for i in probe_ids]
        assert before == after, 'serialized beam inference differs'
        profile = restored.profile(data['state'][probe_ids], data['history16'][probe_ids], data['action'][probe_ids])
        predicted_delta = (design[audit_world] @ ridge + residual.predict(Z[audit_world])) * delta_scale
        result.update(status='fitted', model_sha256=sha(path), final_sha256=sha(path), world_rows=len(iw), event_rows=len(ie),
            actual_hit_rows=int(hit[ie].sum()), actual_natural_end_rows=int(end[ie].sum()),
            source_invalid_rows=int((~data['tail_valid']).sum()), public_schedule_exact=True,
            input_floor=input_floor, delta_floor=delta_floor, ridge_regularization=regularization,
            delta_export_exact_rows=len(audit_world), event_export_exact_rows=len(audit_event),
            serialized_plan_exact_rows=len(probe_ids), inference_profile=profile,
            training_delta_rms_error=float(np.sqrt(np.mean((predicted_delta - delta[audit_world]) ** 2))),
            training_error_is_not_control_evidence=True, tail_sha256=TAIL_SHA)
    except Deadline:
        result.update(status='fit_deadline', no_complete_model=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous_handler)
        result.update(elapsed_seconds=time.monotonic() - began, within_180_seconds=time.monotonic() <= deadline,
            artifact_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        write(output / 'fit-summary.json', result); print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('operation', choices=['prepare', 'fit'])
    parser.add_argument('--output', type=Path, default=DEST); parser.add_argument('--approved-spec-sha256')
    args = parser.parse_args()
    prepare(args.output) if args.operation == 'prepare' else fit(args.output, args.approved_spec_sha256)
