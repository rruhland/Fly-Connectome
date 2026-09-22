"""Preserve the saved event-balanced fit while auditing its temporal shape."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from credit_capacity import score
from fly_connectome.data import checksum


def window_mask(value, fitted):
    if fitted['lower'] is None:
        return np.zeros(len(value), dtype=bool)
    return (value >= fitted['lower']) & (value <= fitted['upper'])


def best_window(value, prediction, target, tempo):
    levels, index = np.unique(value, return_inverse=True)
    groups = [(tempo == d) & (target == s) for d in np.unique(tempo) for s in [-1, 1]]
    quiet = [(tempo == d) & (target == 0) for d in np.unique(tempo)]
    mass = np.column_stack([prediction*target*m for m in groups] +
                           [(abs(prediction) >= .1)*m for m in quiet])
    totals = mass[:, :len(groups)].sum(0)
    assert (totals > 0).all()
    budget = np.floor(.05*np.array([m.sum() for m in quiet]))
    histogram = np.zeros((len(levels), mass.shape[1]))
    np.add.at(histogram, index, mass)
    cumulative = np.vstack([np.zeros(mass.shape[1]), histogram.cumsum(0)])
    best = (0., 0., 0.)
    result = dict(lower=None, upper=None, minimum_retention=0., mean_retention=0.)
    for start in range(len(levels)):
        sums = cumulative[start+1:]-cumulative[start]
        retention = sums[:, :len(groups)]/totals
        alarms = sums[:, len(groups):]
        valid = (alarms <= budget).all(1)
        if not valid.any():
            continue
        low, mean = retention.min(1), retention.mean(1)
        indices = np.flatnonzero(valid)
        order = np.lexsort((-alarms[indices].sum(1), mean[indices], low[indices]))
        chosen = indices[order[-1]]
        key = (float(low[chosen]), float(mean[chosen]), float(-alarms[chosen].sum()))
        if key > best:
            best = key
            end = start+chosen
            lower = levels[start] if start == 0 else (levels[start-1]+levels[start])/2
            upper = levels[end] if end == len(levels)-1 else (levels[end]+levels[end+1])/2
            result = dict(lower=float(lower), upper=float(upper),
                          minimum_retention=key[0], mean_retention=key[1])
    return result


def window_score(value, p, y, fitted):
    gated = p*window_mask(value, fitted)
    result = score(y, gated)
    result['retention'] = {name: float((gated[mask]*y[mask]).sum()/(p[mask]*y[mask]).sum())
        for name, mask in [('ON', y < 0), ('OFF', y > 0)]}
    return result


def main():
    root = Path('runs/credit-capacity-v1')
    source = root/'predictions.npz'
    manifest = json.loads((root/'results.json').read_text())
    digest = checksum(source)
    assert digest == manifest['artifacts'][str(source)]
    data = np.load(source);p = np.clip(data['split-balanced'], -1, 1)
    phases = [];evidence = {}
    for dwell in [2, 3, 4, 6]:
        path = Path(f'runs/context-audit-v1/dwell-{dwell}.npz')
        assert checksum(path) == manifest['reconstruction'][str(dwell)]['sha256']
        original = np.load(path);keep = original['trial'] > 0
        mask = data['tempo'] == dwell
        np.testing.assert_array_equal(data['y'][mask], original['y'][keep])
        phases.extend(original['phase'][keep])
        evidence[str(dwell)] = checksum(path)
    phase = np.asarray(phases);phase_results = {}
    for dwell in [2, 3, 4, 6]:
        test = data['test'] & (data['tempo'] == dwell)
        phase_results[str(dwell)] = {}
        for label in np.unique(phase[test]):
            m = test & (phase == label)
            phase_results[str(dwell)][label] = dict(count=int(m.sum()), mean_prediction=float(p[m].mean()),
                above_threshold=int((abs(p[m]) >= .1).sum()),
                squared_error_sum=float(((p[m]-data['y'][m])**2).sum()))
    value = abs(data['state']);train = data['train']
    noisy = np.maximum(0, value*(1+np.random.default_rng(9092).normal(0, .01, len(value))))
    shared = best_window(value[train], p[train], data['y'][train], data['tempo'][train])
    shared['test'] = {str(d): window_score(value[m], p[m], data['y'][m], shared)
        for d in [2, 3, 4, 6] if (m := data['test'] & (data['tempo'] == d)).any()}
    shared['noisy_test'] = {str(d): window_score(noisy[m], p[m], data['y'][m], shared)
        for d in [2, 3, 4, 6] if (m := data['test'] & (data['tempo'] == d)).any()}
    specific = {}
    for d in [2, 4, 6]:
        m = train & (data['tempo'] == d)
        fitted = best_window(value[m], p[m], data['y'][m], data['tempo'][m])
        test = data['test'] & (data['tempo'] == d)
        fitted['test'] = window_score(value[test], p[test], data['y'][test], fitted)
        fitted['noisy_test'] = window_score(noisy[test], p[test], data['y'][test], fitted)
        specific[str(d)] = fitted
    output = dict(reference_coefficients=manifest['variants']['split-balanced']['coefficients'],
        source_sha256=digest, context_sha256=evidence, phases=phase_results,
        shared_window=shared, tempo_specific_windows=specific,
        sensitivity=dict(seed=9092, relative_standard_deviation=.01, perturbed='window input only'),
        script_sha256=checksum(Path(__file__)))
    out = Path('docs/experiments')
    (out/'2026-09-21-prediction-timing-results.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, preceding, event, sign in [(axes[0], 'off', 'on', -1), (axes[1], 'on', 'off', 1)]:
        for d in [2, 3, 4, 6]:
            row = phase_results[str(d)]
            profile = [sign*row[f'after_{preceding}_{age}']['mean_prediction'] for age in range(1, d)]
            profile += [sign*row[event]['mean_prediction']]
            line, = ax.plot(range(1, d+1), profile, marker='.', label=f'Dwell {d}')
            ax.plot(d, profile[-1], 'o', color=line.get_color(), markerfacecolor='white', markersize=8)
        ax.axhline(.1, color='black', linestyle='--', linewidth=1)
        ax.set(title=f'{event.upper()}-signed forecast', xlabel='Target-frame age since preceding event',
               ylabel='Mean signed amplitude', xticks=range(1, 7))
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Saved split event-balanced fit: open circles mark the due event')
    fig.savefig(out/'assets/2026-09-21-prediction-timing.png', dpi=180)
    assert checksum(source) == digest
    print(json.dumps(dict(shared=shared, specific=specific), indent=2), flush=True)


if __name__ == '__main__':
    main()
