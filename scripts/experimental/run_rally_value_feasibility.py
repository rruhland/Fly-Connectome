"""Prepare/run one saved-TRAIN LSPI feasibility fit. Never executes cameras."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from pong_fixed_rbf import load_readout
from pong_fixed_pairwise import FrozenPairwiseReadout, save_readout
from pong_value_lspi import fit_lspi, map_weights

ROOT = Path('/workspace/m1b-experiments')
DEST = ROOT/'pong-value-feasibility-20261028-r1'
DATA = [ROOT/'pong-rally-full-actor-20261028-r1',
        ROOT/'pong-rally-smooth-actor-20261028-r1']
BASE = ROOT/'pong-rally-joint-output-20261028-r1/full'
BANK = ROOT/'pong-rally-TRAIN-bank-20261028-r7/admission.json'
BASIS = DATA[1]/'correction-basis.npy'
EXPECTED = '40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def prepare():
    assert not DEST.exists()
    DEST.mkdir()
    admitted = {r['seed'] for r in read(BANK)['bank']}
    files = []
    for source in DATA:
        declared = read(source/'training-schedule.json')['batches']
        jobs = {(j['batch'], j['episode']):j for batch in declared for j in batch}
        for path in sorted((source/'training').glob('batch_*_episode_*.json')):
            receipt = read(path)
            assert receipt['seed'] in admitted
            job = jobs[(receipt['batch'], receipt['episode'])]
            assert job['seed'] == receipt['seed'] and job['action_seed'] == receipt['action_seed']
            assert receipt['report']['config']['action_hold_frames'] == 1
            assert receipt['report']['config']['episodes'] == 1
            files.append(dict(receipt=str(path), seed=receipt['seed'],
                              transitions=receipt['transitions'], complete=receipt['complete']))
    spec = dict(method='three_fixed_LSPI_LSTDQ_iterations',
        source_training_roots=[str(p) for p in DATA], episodes=files,
        base=str(BASE), baseline_bundle_sha256=EXPECTED,
        bank=str(BANK), bank_sha256=sha(BANK), basis=str(BASIS), basis_sha256=sha(BASIS),
        feature_state_dimension=711, action_dimension=2133,
        feature_definition='DCT198 + existing capped frame-present raw metadata coordinate4384 + existing fixed512 TensorSketch',
        normalization='TRAIN weighted second-moment RMS only; floor0.1medianpositive; absorbed into mapped W/R',
        iterations=3, ridge=.01, gamma_per_second=.95, fit_seconds=120,
        sample_seed=2026102881, max_rows_per_episode=46, max_total_rows=12000,
        sampling='all available natural-terminal rows plus uniform remaining rows; inverse inclusion probability and equal episode weights',
        last_row='natural termination zero bootstrap; drop unavailable next observation on timeout or partial data',
        duration='frozen camera 12/5 tick schedule at 1/120s; verify saved total seconds, last tick count may shorten',
        selection='FINAL third completed iteration, no sweep/checkpoint selection',
        target_policy_initial='greedy40069 only; its scores are NEVER Q-value targets',
        no_new_runtime_architecture=True, no_native_cameras=True, no_fresh_data=True,
        no_automatic_promotion=True, approval_required_before_fit=True,
        source_files={str(Path(__file__)):sha(__file__),
                      str(Path(__file__).with_name('pong_value_lspi.py')):sha(Path(__file__).with_name('pong_value_lspi.py'))})
    write(DEST/'spec.json', spec)
    print(json.dumps(dict(status='prepared_no_fit', episodes=len(files),
                         possible_transitions=sum(f['transitions'] for f in files),
                         complete_episodes=sum(f['complete'] for f in files), spec_sha256=sha(DEST/'spec.json'))))


def load_data(spec, baseline, B):
    rng = np.random.default_rng(spec['sample_seed'])
    rows, nextrows, actions, rewards, discounts, weights, probs, traces = [], [], [], [], [], [], [], []
    skipped = []
    for item in spec['episodes']:
        receipt_path = Path(item['receipt'])
        receipt = read(receipt_path)
        raw_path = Path(receipt['raw_feature_file'])
        scores_path = Path(receipt['scores_file'])
        assert raw_path.parent == receipt_path.parent and scores_path.parent == receipt_path.parent
        X = sp.load_npz(raw_path).tocsr()
        with np.load(scores_path, allow_pickle=False) as scores:
            act = scores['actions'].copy()
            hits = scores['hits'].copy()
            p = scores['p'].copy()
        n = len(act)
        assert X.shape == (n, 4389) and hits.shape == (n,) and p.shape == (n,3)
        assert n == receipt['transitions'] and int(hits.sum()) == receipt['hits']
        assert np.isin(act, [0,1,2]).all() and np.isin(hits, [0,1]).all()
        assert np.isfinite(X.data).all() and np.isfinite(p).all() and np.all(p >= 0)
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-14)
        ticks = np.diff(np.arange(n+1, dtype=np.int64)*12//5)
        report = receipt['report']
        complete = bool(report['complete'])
        if complete:
            rally = report['rallies'][0]
            seconds = float(rally['seconds'])
            total_ticks = round(seconds*120)
            assert abs(seconds-total_ticks/120) < 1e-9
            last_ticks = total_ticks-int(ticks[:-1].sum())
            assert 1 <= last_ticks <= ticks[-1]
            ticks[-1] = last_ticks
            terminated = bool(rally['terminated'])
        else:
            terminated = False
            assert abs(report['physics_seconds']-ticks.sum()/120) < 1e-9
        valid = n if terminated else n-1
        if valid <= 0:
            skipped.append(str(receipt_path))
            continue
        last = valid-1
        m = min(spec['max_rows_per_episode'], valid)
        # Endpoint stratification is outcome-independent within each observed episode.
        sampled = np.sort(np.r_[rng.choice(last, m-1, replace=False), last]) if m>1 else np.array([last])
        probability = np.full(m, (m-1)/last if last else 1.)
        probability[sampled == last] = 1.
        w = 1./(valid*probability)
        nxt = np.minimum(sampled+1, n-1)
        terminal = terminated & (sampled == n-1)
        raw = X[sampled]
        nextraw = X[nxt].copy()
        if np.any(terminal):
            nextraw = nextraw.tolil()
            nextraw[np.flatnonzero(terminal)] = 0.
            nextraw = nextraw.tocsr()
        rows.append(raw); nextrows.append(nextraw)
        actions.append(act[sampled]); rewards.append(hits[sampled].astype(float))
        disc = spec['gamma_per_second']**(ticks[sampled]/120.)
        disc[terminal] = 0.
        discounts.append(disc); weights.append(w); probs.append(p[sampled])
        traces.append(dict(seed=receipt['seed'], receipt=str(receipt_path),
            receipt_sha256=sha(receipt_path), raw_sha256=sha(raw_path), scores_sha256=sha(scores_path),
            rows=sampled.tolist(), selected_rows=m, available_rows=valid,
            natural_terminal=terminated, dropped_unobserved_next_row=not terminated,
            reward_events=int(hits[sampled].sum()), episode_seconds=report['physics_seconds']))
    X = sp.vstack(rows, format='csr'); Y = sp.vstack(nextrows, format='csr')
    assert len(X.indptr)-1 <= spec['max_total_rows']
    F = np.column_stack((X@B.T, baseline.basis.batch(X)))
    N = np.column_stack((Y@B.T, baseline.basis.batch(Y)))
    w = np.concatenate(weights)/len(weights)
    assert np.isclose(w.sum(), 1.)
    rms = np.sqrt(np.sum(w[:,None]*F*F, axis=0))
    positive = rms[rms>0]
    floor = .1*float(np.median(positive))
    denominator = np.maximum(rms, floor)
    # At terminal transitions N is irrelevant: keep it exactly zero for clarity.
    discount = np.concatenate(discounts)
    N[discount == 0] = 0.
    initial_q = np.asarray(Y@baseline.base.weights.T) + baseline.basis.batch(Y)@baseline.residual.T
    target = np.argmax(initial_q, axis=1)
    return dict(X=X, F=F/denominator, N=N/denominator,
        denominator=denominator, actions=np.concatenate(actions), rewards=np.concatenate(rewards),
        discounts=discount, weights=w, probabilities=np.vstack(probs), initial_policy=target,
        traces=traces, skipped=skipped, rms_floor=floor)


def run(approval):
    spec_path = DEST/'spec.json'
    assert approval == sha(spec_path), 'root approval must identify the exact prepared spec hash'
    assert not (DEST/'work-clock.json').exists(), 'one fit only; no restart'
    spec = read(spec_path)
    began = time.monotonic(); deadline = began+spec['fit_seconds']
    write(DEST/'work-clock.json', dict(started_wall_ns=time.time_ns(), started_monotonic=began,
        deadline_monotonic=deadline, seconds=spec['fit_seconds'], new_cameras=0))
    assert all(sha(p)==h for p,h in spec['source_files'].items())
    assert sha(BANK)==spec['bank_sha256'] and sha(BASIS)==spec['basis_sha256']
    assert sha(BASE/'rbf-model.npz')==EXPECTED
    baseline = load_readout(BASE/'rbf-model.npz')
    existing = np.load(BASIS, allow_pickle=False)
    assert existing.shape == (198,4389)
    B = np.zeros((199,4389)); B[:198] = existing; B[198,4384] = 1.
    data = load_data(spec, baseline, B)
    write(DEST/'data-manifest.json', dict(traces=data['traces'], skipped=data['skipped'],
        selected_rows=len(data['F']), state_features=data['F'].shape[1],
        weighted_reward=float(data['weights']@data['rewards']), reward_rows=int(data['rewards'].sum()),
        actions=np.bincount(data['actions'], minlength=3).tolist(), rms_floor=data['rms_floor'],
        action_probabilities='saved behavior categorical probabilities; report extrapolation only',
        cross_episode_edges=0, fresh_rows=0, new_cameras=0))
    print(json.dumps(dict(phase='data_prepared', rows=len(data['F']), reward_rows=int(data['rewards'].sum()), elapsed=time.monotonic()-began)), flush=True)
    fit_began = time.monotonic()
    theta, history = fit_lspi(data['F'], data['N'], data['actions'], data['rewards'],
        data['discounts'], data['weights'], data['initial_policy'],
        iterations=spec['iterations'], ridge=spec['ridge'], deadline=deadline)
    W,R = map_weights(theta, data['denominator'], B)
    candidate = FrozenPairwiseReadout(W, baseline.basis, R)
    # Check actual existing deployed sparse arithmetic on every selected input.
    q_expected = data['F']@theta.T
    q_batch = np.asarray(data['X']@W.T)+baseline.basis.batch(data['X'])@R.T
    np.testing.assert_allclose(q_batch, q_expected, rtol=1e-9, atol=1e-9)
    errors=[]; deployed_actions=[]
    for i in range(len(data['F'])):
        lo,hi=data['X'].indptr[i:i+2]
        action,q=candidate.choose(data['X'].indices[lo:hi], data['X'].data[lo:hi])
        errors.append(float(np.max(np.abs(q-q_expected[i])))); deployed_actions.append(action)
    deployed_actions=np.asarray(deployed_actions)
    assert np.array_equal(deployed_actions, np.argmax(q_expected,axis=1))
    np.save(DEST/'proposal.npy', W); save_readout(DEST/'rbf-model.npz',W,baseline.basis,R)
    np.save(DEST/'theta.npy',theta); np.save(DEST/'feature-denominator.npy',data['denominator'])
    np.save(DEST/'raw-basis.npy',B)
    (DEST/'radial-cap.json').write_bytes((BASE/'radial-cap.json').read_bytes())
    baseq=np.asarray(data['X']@baseline.base.weights.T)+baseline.basis.batch(data['X'])@baseline.residual.T
    chosen_p=data['probabilities'][np.arange(len(deployed_actions)),deployed_actions]
    summary=dict(status='zero_native_fit_complete',new_transition_cameras=0,new_initial_cameras=0,
        fit_seconds=time.monotonic()-fit_began,wall_seconds=time.monotonic()-began,
        history=history,complete_fixed_iterations=len(history),candidate_bundle_sha256=sha(DEST/'rbf-model.npz'),
        W_sha256=sha(DEST/'proposal.npy'),same_runtime_schema=candidate.schema==baseline.schema,
        mapped_batch_maximum_error=float(np.max(np.abs(q_batch-q_expected))),
        mapped_deployed_maximum_error=max(errors),
        greedy_changes_from40069=int(np.count_nonzero(deployed_actions!=np.argmax(baseq,axis=1))),
        learned_action_counts=np.bincount(deployed_actions,minlength=3).tolist(),
        minimum_behavior_probability=float(chosen_p.min()),
        selected_action_behavior_probability_quantiles=np.quantile(chosen_p,[0,.01,.1,.5,1]).tolist(),
        chosen_behavior_probability_below01=int(np.count_nonzero(chosen_p<.01)),
        chosen_behavior_probability_below05=int(np.count_nonzero(chosen_p<.05)),
        no_backprop=True,no_baseline_q_value_targets=True,no_automatic_promotion=True,
        native_qualification='NONE; root review required',
        limitation='off-policy LSTDQ and sparse alternative-action coverage; fitted Bellman residual is not policy improvement')
    assert time.monotonic()<=deadline, 'completed after bound; do not qualify as complete fit'
    write(DEST/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','fit'])
    parser.add_argument('--approved-spec-sha256');args=parser.parse_args()
    if args.operation=='prepare':prepare()
    else:run(args.approved_spec_sha256)
