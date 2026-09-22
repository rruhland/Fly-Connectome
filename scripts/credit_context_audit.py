"""Offline per-edge credit-context probe on saved frozen baseline states.

No model training or new state. Updates are counterfactual proposals at frozen
weights, with the common positive learning rate omitted, not a training replay.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from temporal_visual import recurrent_boundaries


def balanced(y, predicted):
    return float(np.mean([(predicted[y == s] == s).mean() for s in [-1, 1]]))


def stump(x, y, sample_weight=None):
    best = None
    for col in range(x.shape[1]):
        values = np.unique(x[:, col])
        thresholds = np.r_[np.nextafter(values[0], -np.inf),
                           (values[:-1] + values[1:]) / 2, values[-1]]
        for threshold in thresholds:
            predicted = np.where(x[:, col] > threshold, 1, -1)
            score = (balanced(y, predicted) if sample_weight is None else
                     float(sample_weight[y == predicted].sum()/sample_weight.sum()))
            polarity = 1 if score >= .5 else -1
            key = max(score, 1-score)
            if best is None or key > best[0]:
                best = (key, col, float(threshold), polarity)
    return best


def metrics(update, target, predicted):
    sign = np.sign(update)
    absolute = np.abs(update).sum()
    return dict(count=len(update), balanced_sign_accuracy=balanced(sign, predicted),
        update_mass_correct=float(np.abs(update)[sign == predicted].sum()/absolute),
        scalar_retention=float(abs(update.sum())/absolute),
        two_context_retention=float(sum(abs(update[predicted == s].sum()) for s in [-1, 1])/absolute),
        categories={name: dict(count=int(mask.sum()),
            sign_accuracy=float((sign[mask] == predicted[mask]).mean()),
            positive=int((update[mask] > 0).sum()), negative=int((update[mask] < 0).sum()),
            sum=float(update[mask].sum()), absolute_sum=float(np.abs(update[mask]).sum()))
            for name, mask in [('ON', target < 0), ('OFF', target > 0), ('quiet', target == 0)]})


def main():
    source = Path('runs/context-audit-v1')
    datasets = []
    hashes = {}
    original = json.loads((source/'results.json').read_text())
    for dwell in [2, 3, 4, 6]:
        path = source/f'dwell-{dwell}.npz'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[str(path)] == original['artifacts'][str(path)]
        a = np.load(path)
        issue = recurrent_boundaries(a['blank_frames'], dwell=dwell)
        np.testing.assert_array_equal(a['y'], a['all_target'][issue+8])
        np.testing.assert_array_equal(a['prediction'], a['all_prediction'][issue])
        names = a['feature_names'].tolist()
        datasets.append({**{k: a[k] for k in ['x', 'y', 'trial', 'prediction']},
                         'tempo': np.full(len(a['y']), dwell)})
    data = {k: np.concatenate([a[k] for a in datasets]) for k in datasets[0]}
    results = {}
    for pre in [20655, 26550]:
        eligibility = data['x'][:, names.index(f'eligibility_{pre}')]
        update = (data['y']-data['prediction'])*eligibility
        active = update != 0
        train = active & (data['tempo'] != 3) & (data['trial'] < 15)
        # Original calibration trials remain unused: no tuning on them or test.
        test = active & ((data['tempo'] == 3) | (data['trial'] >= 20))
        labels = np.sign(update[train])
        groups = {'EI_eligibility': [1, 2, names.index(f'eligibility_{pre}')],
                  'edge_and_post': list(range(14)) + [names.index(f'{s}_{pre}')
                      for s in ['eligibility', 'arrival_trace', 'arrival_now']]}
        edge = {'zero_update_count': int((~active).sum()), 'groups': {}}
        for name, cols in groups.items():
            x = data['x'][:, cols]
            _, col, threshold, polarity = stump(x[train], labels)
            simple = polarity*np.where(x[test, col] > threshold, 1, -1)
            alternatives = {}
            boundaries = {}
            categories = np.sign(data['y'][train])
            event_weights = np.asarray([1/(categories == c).sum() for c in categories])
            for objective, weights in [('event_balanced', event_weights),
                                       ('update_mass', np.abs(update[train]))]:
                _, j, boundary, direction = stump(x[train], labels, weights)
                alternatives[objective] = direction*np.where(x[test, j] > boundary, 1, -1)
                boundaries[objective] = dict(feature=names[cols[j]], threshold=boundary, polarity=direction)
            mean, scale = x[train].mean(0), x[train].std(0)
            keep = scale > 1e-8
            z = (x[:, keep]-mean[keep])/scale[keep]
            _, neighbors = cKDTree(z[train]).query(z[test], k=15)

            def vote(y):
                counts = {s: (y == s).sum() for s in [-1, 1]}
                votes = y[neighbors]
                return np.where((votes == 1).sum(1)/counts[1] >
                                (votes == -1).sum(1)/counts[-1], 1, -1)

            predicted = vote(labels)
            if name == 'edge_and_post':
                # Exploratory structural comparator added after the fitted stump
                # selected sensory_state; no fitted threshold or model gate.
                direction = -1 if pre == 20655 else 1
                alternatives['sensory_zero'] = direction*np.where(data['x'][test, 3] > 0, 1, -1)
            scores = {}
            for dwell in [2, 3, 4, 6]:
                mask = data['tempo'][test] == dwell
                scores[str(dwell)] = {kind: metrics(update[test][mask], data['y'][test][mask], p[mask])
                                     for kind, p in [('stump', simple), ('knn', predicted), *alternatives.items()]}
            rng = np.random.default_rng(9090)
            held = data['tempo'][test] == 3
            null = [balanced(np.sign(update[test][held]), vote(rng.permutation(labels))[held])
                    for _ in range(20)]
            edge['groups'][name] = dict(features=[names[i] for i in cols],
                stump=dict(feature=names[cols[col]], threshold=threshold, polarity=polarity),
                alternative_boundaries=boundaries,
                scores=scores, held3_knn_shuffled_balanced_accuracy=null)
        results[str(pre)] = edge
    out = Path('runs/credit-context-audit-v1')
    out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(dict(source_hashes=hashes, results=results), indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
