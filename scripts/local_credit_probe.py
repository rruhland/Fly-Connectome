"""Causal local amplitude-credit replay on saved frozen histories."""
import json
from pathlib import Path

import numpy as np

from credit_capacity import design, score
from temporal_visual import recurrent_boundaries
from fly_connectome.data import checksum


def issue_design(eligibility, sensory, split):
    return design(np.asarray(eligibility), np.asarray(sensory), split)


def step(weights, eligibility, target, gate, gate_credit, eta, gain):
    active = bool(gate) if gate_credit else True
    prediction = float(np.clip(eligibility@weights, -1, 1)) if active else 0.
    update = eta*gain*(target-prediction)*eligibility if active else 0.
    return np.clip(weights+update, 0, 10), prediction, gain


def replay(x, y, gate, initial, eta, *, gate_credit=True, balanced=False, passes=1):
    weights = np.asarray(initial, dtype=float).copy()
    counts = np.ones(2)
    first_gains = np.zeros(len(y))
    first_issued = np.zeros(len(y))
    checkpoints = {}
    for epoch in range(passes):
        for i in range(len(y)):
            gain = min(8., max(1., counts[0]/counts[1])) if balanced and y[i] != 0 else 1.
            weights, prediction, _ = step(weights, x[i], y[i], gate[i], gate_credit, eta, gain)
            if epoch == 0:
                first_gains[i] = gain
                first_issued[i] = prediction if gate[i] else 0.
            counts[int(y[i] != 0)] += 1
        checkpoints[str(epoch+1)] = weights.copy()
    return dict(weights=weights, checkpoints=checkpoints, gains=first_gains,
                first_issued=first_issued, counts=counts)


def metrics(x, y, gate, weights):
    raw = x@weights
    return dict(gated=score(y, np.clip(raw, -1, 1)*gate),
                ungated=score(y, raw))


def choose(candidates):
    best = None
    for eta, item in candidates.items():
        m = item['validation']['gated']
        anticipation = min(m['on_anticipation'], m['off_anticipation'])
        if m['false_alarm_fraction'] <= .05:
            passing = anticipation >= .1
            key = (2 if passing else 1, -m['mse'] if passing else anticipation,
                   anticipation if passing else -m['mse'], -float(eta))
        else:
            key = (0, -m['false_alarm_fraction'], anticipation, -m['mse'])
        if best is None or key > best[0]:
            best = key, eta
    return best[1]


def load():
    capacity = Path('runs/credit-capacity-v1/predictions.npz')
    manifest = json.loads(capacity.with_name('results.json').read_text())
    assert checksum(capacity) == manifest['artifacts'][str(capacity)]
    timing_path = Path('runs/causal-timing-v1/replay.npz')
    timing_manifest = json.loads(Path('docs/experiments/2026-09-22-causal-timing-results.json').read_text())
    assert checksum(timing_path) == timing_manifest['replay_sha256']
    d = np.load(capacity); timing = np.load(timing_path)
    gate_parts = []
    for dwell in [2, 3, 4, 6]:
        path = Path(f'runs/context-audit-v1/dwell-{dwell}.npz')
        assert checksum(path) == manifest['reconstruction'][str(dwell)]['sha256']
        original = np.load(path)
        issue = recurrent_boundaries(original['blank_frames'], dwell=dwell)//8
        keep = original['trial'] > 0
        issue = issue[keep]
        mask = d['tempo'] == dwell
        np.testing.assert_array_equal(original['y'][keep], d['y'][mask])
        np.testing.assert_array_equal(original['x'][keep, 3], d['state'][mask])
        gate_parts.append(timing[f'dwell{dwell}_gate'][issue])
    gate = np.concatenate(gate_parts)
    weights_table = json.loads(Path('docs/experiments/2026-09-21-short-term-learning-weights.json').read_text())
    assert [row['pre'] for row in weights_table] == manifest['presynaptic_bodies']
    initial = np.array([row['initial'] for row in weights_table])
    challenges = {}
    continuous = json.loads(Path('docs/experiments/2026-09-22-continuous-timing-results.json').read_text())
    for kind in ['standard', 'omitted']:
        path = Path(f'runs/continuous-timing-v1/{kind}.npz')
        assert checksum(path) == continuous['artifacts'][str(path)]
        a = np.load(path)
        np.testing.assert_array_equal(a['presynaptic_bodies'], manifest['presynaptic_bodies'])
        challenges[kind] = dict(x=a['eligibility'][:-1], sensory=a['state'][:-1],
                                gate=a['gate'][:-1], y=a['events'][1:])
    return d, gate, initial, challenges, dict(capacity=checksum(capacity),
        timing=checksum(timing_path), continuous=checksum(Path('docs/experiments/2026-09-22-continuous-timing-results.json')))


def main():
    d, gate, initial, challenges, sources = load()
    y, state = d['y'], d['state']
    train = d['train']
    validation = (d['tempo'] != 3) & (d['trial'] >= 15) & (d['trial'] < 20)
    tests = {'known': d['test'] & (d['tempo'] != 3), 'unseen3': d['tempo'] == 3}
    tests.update({f'tempo{t}': d['test'] & (d['tempo'] == t) for t in [2, 4, 6]})
    variants = {
        'shared-original': dict(split=False, gate_credit=False, balanced=False),
        'shared-gate-consistent': dict(split=False, gate_credit=True, balanced=False),
        'shared-gate-balanced': dict(split=False, gate_credit=True, balanced=True),
        'split-ungated-credit': dict(split=True, gate_credit=False, balanced=False),
        'split-gate-consistent': dict(split=True, gate_credit=True, balanced=False),
        'split-gate-balanced': dict(split=True, gate_credit=True, balanced=True),
    }
    result = dict(sources=sources, train_count=int(train.sum()), validation_count=int(validation.sum()),
        tests={k:int(v.sum()) for k,v in tests.items()}, gate_train_open=int(gate[train].sum()),
        eta_grid=[.01,.03,.1,.3,1.], passes=[1,5], variants={},
        offline_reference={k:score(y[m], np.clip(d['split-balanced'][m], -1, 1)*gate[m])
                           for k,m in tests.items()})
    for name, spec in variants.items():
        x = issue_design(d['x'], state, spec['split'])
        w0 = np.tile(initial, 2) if spec['split'] else initial
        row = dict(initial={k:metrics(x[m], y[m], gate[m], w0) for k,m in tests.items()},
                   candidates={})
        for eta in result['eta_grid']:
            run = replay(x[train], y[train], gate[train], w0, eta,
                         gate_credit=spec['gate_credit'], balanced=spec['balanced'], passes=5)
            candidate = dict(first_pass=dict(weights=run['checkpoints']['1'].tolist(),
                validation=metrics(x[validation], y[validation], gate[validation], run['checkpoints']['1'])),
                prequential={label:score(y[train][mask], run['first_issued'][mask])
                    for label, mask in [('all', np.ones(train.sum(), dtype=bool)),
                        ('tempo2', d['tempo'][train] == 2),
                        ('tempo4', d['tempo'][train] == 4),
                        ('tempo6', d['tempo'][train] == 6)]},
                weights=run['weights'].tolist(),
                validation=metrics(x[validation], y[validation], gate[validation], run['weights']),
                local_counts=run['counts'].tolist(),
                event_gain=dict(min=float(run['gains'][y[train]!=0].min()),
                                max=float(run['gains'][y[train]!=0].max()),
                                mean=float(run['gains'][y[train]!=0].mean())))
            row['candidates'][str(eta)] = candidate
        selected = choose(row['candidates']); weights = np.asarray(row['candidates'][selected]['weights'])
        row['selected_eta'] = selected
        row['test'] = {k:metrics(x[m], y[m], gate[m], weights) for k,m in tests.items()}
        row['continuous'] = {kind:metrics(issue_design(a['x'], a['sensory'], spec['split']),
                                           a['y'], a['gate'], weights) for kind,a in challenges.items()}
        result['variants'][name] = row
        print(name, 'eta', selected,
              'validation', row['candidates'][selected]['validation']['gated'],
              'unseen3', row['test']['unseen3']['gated'], flush=True)
    result['script_sha256'] = checksum(Path(__file__))
    output = Path('docs/experiments/2026-09-22-local-credit-results.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
