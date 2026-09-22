"""Offline constrained fits on frozen local histories; never installs weights."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from temporal_visual import recurrent_boundaries
from fly_connectome.data import checksum


def design(eligibility, sensory, split):
    if not split:
        return eligibility.copy()
    positive = sensory > 0
    return np.concatenate([eligibility*(~positive[:, None]), eligibility*positive[:, None]], axis=1)


def fit(x, y, maximum, balanced):
    mass = np.ones(len(y))
    if balanced:
        for mask in [y < 0, y > 0, y == 0]:
            mass[mask] = 1/mask.sum()
    mass /= mass.mean()
    active = np.any(x != 0, axis=0)
    result = lsq_linear(x[:, active]*np.sqrt(mass[:, None]), y*np.sqrt(mass),
                        bounds=(0, maximum), method='bvls', tol=1e-12)
    assert result.success
    weights = np.zeros(x.shape[1]);weights[active] = result.x
    return weights, dict(success=result.success, optimality=float(result.optimality),
        iterations=result.nit, cost=float(result.cost), active_columns=np.flatnonzero(active).tolist())


def score(y, raw):
    p = np.clip(raw, -1, 1)
    quiet = y == 0
    output = dict(samples=len(y), mse=float(np.mean((y-p)**2)),
        quiet_mse=float(np.mean(p[quiet]**2)),
        false_alarm_fraction=float(np.mean(np.abs(p[quiet]) >= .1)),
        clipped_samples=int((raw != p).sum()))
    for name, mask in [('on', y < 0), ('off', y > 0)]:
        output[name+'_count'] = int(mask.sum())
        output[name+'_mse'] = float(np.mean((y[mask]-p[mask])**2))
        output[name+'_anticipation'] = float(np.mean(y[mask]*p[mask]))
    output['anticipation_and_quiet_pass'] = bool(output['on_anticipation'] >= .1 and
        output['off_anticipation'] >= .1 and output['false_alarm_fraction'] <= .05)
    return output


def main():
    source = Path('runs/context-audit-v1')
    original = json.loads((source/'results.json').read_text())
    weight_source = Path('runs/multitempo-v1/mixed-training.npz')
    assert checksum(weight_source) == original['mixed_weights_sha256']
    weights_table = json.loads(Path('docs/experiments/2026-09-21-short-term-learning-weights.json').read_text())
    old = np.load(weight_source)
    magnitudes = old['weights_trained'][old['incoming_edges']]
    np.testing.assert_array_equal(magnitudes, [row['baseline'] for row in weights_table])
    maximum = json.loads(Path('runs/temporal-area-matched-excitation-v1/results.json').read_text())['manifest']['learning']['maximum_weight']
    parts, evidence = [], {}
    for dwell in [2, 3, 4, 6]:
        path = source/f'dwell-{dwell}.npz'
        assert checksum(path) == original['artifacts'][str(path)]
        a = np.load(path);names = a['feature_names'].tolist()
        cols = [names.index(f'eligibility_{row["pre"]}') for row in weights_table]
        issued = recurrent_boundaries(a['blank_frames'], dwell=dwell)
        np.testing.assert_array_equal(a['y'], a['all_target'][issued+8])
        np.testing.assert_array_equal(a['prediction'], a['all_prediction'][issued])
        keep = a['trial'] > 0
        x, y = a['x'][keep][:, cols], a['y'][keep]
        reconstructed = x@magnitudes
        np.testing.assert_allclose(reconstructed, a['prediction'][keep], atol=1e-6, rtol=0)
        assert all((v >= 0).all() or (v <= 0).all() for v in x.T)
        evidence[str(dwell)] = dict(sha256=checksum(path),
            reconstruction_max_error=float(abs(reconstructed-a['prediction'][keep]).max()))
        parts.append(dict(x=x, y=y, state=a['x'][keep, 3], trial=a['trial'][keep],
            tempo=np.full(len(y), dwell), recorded=a['prediction'][keep],
            persistence=a['all_target'][issued][keep]))
    data = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    train = (data['tempo'] != 3) & (data['trial'] < 15)
    test = (data['tempo'] == 3) | (data['trial'] >= 20)
    masks = {f'test{d}': test & (data['tempo'] == d) for d in [2, 3, 4, 6]}
    masks['train'] = train
    variants = {name: dict(scores={label: score(data['y'][mask], prediction[mask])
                     for label, mask in masks.items()})
                for name, prediction in [('recorded', data['recorded']), ('zero', np.zeros(len(train))),
                                         ('persistence', data['persistence'])]}
    arrays = {}
    for split in [False, True]:
        matrix = design(data['x'], data['state'], split)
        for balanced in [False, True]:
            name = ('split' if split else 'shared')+('-balanced' if balanced else '-mse')
            coefficients, optimizer = fit(matrix[train], data['y'][train], maximum, balanced)
            raw = matrix@coefficients
            variants[name] = dict(coefficients=coefficients.tolist(), optimizer=optimizer,
                scores={label: score(data['y'][mask], raw[mask]) for label, mask in masks.items()})
            arrays[name] = raw
    out = Path('runs/credit-capacity-v1');out.mkdir(exist_ok=True)
    path = out/'predictions.npz'
    np.savez_compressed(path, **arrays, **data, train=train, test=test)
    report = dict(maximum_weight=maximum, reconstruction=evidence,
        presynaptic_bodies=[row['pre'] for row in weights_table], variants=variants,
        artifacts={str(path): checksum(path), str(weight_source): checksum(weight_source)},
        script_sha256=checksum(Path(__file__)))
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for name, row in variants.items():
        print(name, json.dumps(row['scores']['test3']), flush=True)


if __name__ == '__main__':
    main()
