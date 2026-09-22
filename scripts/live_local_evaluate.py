"""Held-out neural feedback tests for the selected live local learner."""
import json
from pathlib import Path

import numpy as np
import torch

from continuous_timing import challenge, summarize
from live_local_timing import OUT, report_run, run_frames, score_run, starting_weights
from temporal_visual import oscillation, recurrent_boundaries
from timing_transfer import load_model
from fly_connectome.data import checksum


def safe_score(y, p):
    result = summarize(y, p)
    for name, mask in [('on', y < 0), ('off', y > 0), ('quiet', y == 0)]:
        result[name+'_mse'] = float(np.mean((y[mask]-p[mask])**2)) if mask.any() else None
    return result


def continuous_scores(data, times):
    frames = np.arange(1, len(data['target']))
    y, p, persistence = data['target'][1:], data['gated'][:-1], data['target'][:-1]
    masks = {'all': np.ones(len(y), dtype=bool),
             'startup_quiet': frames < times[0], 'trailing_quiet': frames > times[-1]}
    for block, dwell in enumerate([2, 3, 6, 4]):
        begin = times[block*24]
        end = times[(block+1)*24] if block < 3 else times[-1]+1
        within = (frames >= begin) & (frames < end)
        recovery = (frames >= times[60]) & (frames < times[64]) if block == 2 else np.zeros(len(y), bool)
        masks[f'dwell{dwell}_all'] = within
        masks[f'dwell{dwell}_adaptation'] = within & (frames < times[block*24+4])
        masks[f'dwell{dwell}_steady'] = within & (frames >= times[block*24+4]) & ~recovery
        if block == 2:
            masks['omission_recovery'] = recovery
    result = {}
    for name, mask in masks.items():
        result[name] = dict(count=int(mask.sum()), events=int((y[mask]!=0).sum()),
            learned=safe_score(y[mask], p[mask]),
            persistence=safe_score(y[mask], persistence[mask]))
    return result


def save(path, result):
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **result)
    return checksum(path)


def main():
    torch.set_num_threads(1)
    crop, metadata, _ = load_model()
    selection_path = OUT/'train.json'
    selected = json.loads(selection_path.read_text())
    if selected['selected_eta'] not in selected['candidates']:
        raise ValueError('training selection incomplete')
    source = Path('runs/multitempo-v1/mixed-training.npz')
    if checksum(source) != selected['source_sha256']:
        raise ValueError('source weights changed')
    initial = starting_weights(crop, torch.from_numpy(np.load(source)['weights_trained']))
    train_path = OUT/f"train-eta-{selected['selected_eta']}.npz"
    if checksum(train_path) != selected['artifacts'][str(train_path)]:
        raise ValueError('selected training artifact changed')
    trained_record = np.load(train_path)
    trained = torch.from_numpy(trained_record['weights_trained'])
    edge = trained_record['incoming_edges']
    outside = np.ones(len(initial), bool); outside[edge] = False
    np.testing.assert_array_equal(trained.numpy()[outside], initial.numpy()[outside])
    previous = np.vstack([initial.numpy()[edge], trained_record['weights'][:-1]])
    proposed = np.clip(previous+trained_record['delta'], 0, 10)
    np.testing.assert_allclose(proposed, trained_record['weights'], rtol=1e-6, atol=1e-6)
    report = dict(training_sha256=checksum(train_path), selection_sha256=checksum(selection_path),
        initial_target_weights=initial.numpy()[edge].tolist(),
        trained_target_weights=trained.numpy()[edge].tolist(),
        audit_max_abs=float(abs(proposed-trained_record['weights']).max()),
        source_sha256=checksum(source), evaluations={}, artifacts={})
    blanks = np.random.default_rng(9172).integers(12, 37, size=8)
    report['heldout_blanks'] = blanks.tolist()
    for dwell in [2, 3, 4, 6]:
        frames = torch.cat([oscillation(int(blank), dwell=dwell) for blank in blanks])
        issue = recurrent_boundaries(blanks, dwell=dwell)//8
        results = {}
        for label, weights in [('initial', initial), ('trained', trained)]:
            r = run_frames(crop, metadata, weights, frames, eta=0.)
            results[label] = r
            path = OUT/f'heldout-{label}-dwell{dwell}.npz'
            report['artifacts'][str(path)] = save(path, r)
        np.testing.assert_array_equal(results['initial']['target'], results['trained']['target'])
        np.testing.assert_array_equal(results['initial']['gate'], results['trained']['gate'])
        report['evaluations'][f'dwell{dwell}'] = {label:report_run(r, issue, metadata['neurons']['dt'])
                                                  for label,r in results.items()}
        m = report['evaluations'][f'dwell{dwell}']
        print('dwell', dwell, 'initial', json.dumps(m['initial']['scores']['learned']),
              'trained', json.dumps(m['trained']['scores']['learned']), flush=True)
        (OUT/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    old_manifest = json.loads(Path('docs/experiments/2026-09-22-continuous-timing-results.json').read_text())
    for kind in ['standard', 'omitted']:
        frames, times, _ = challenge(kind == 'omitted')
        results = {}
        for label, weights in [('initial', initial), ('trained', trained)]:
            r = run_frames(crop, metadata, weights, frames, eta=0.)
            results[label] = r
            path = OUT/f'continuous-{label}-{kind}.npz'
            report['artifacts'][str(path)] = save(path, r)
        old_path = Path(f'runs/continuous-timing-v1/{kind}.npz')
        assert checksum(old_path) == old_manifest['artifacts'][str(old_path)]
        old = np.load(old_path)
        for r in results.values():
            np.testing.assert_array_equal(r['target'], old['events'])
        np.testing.assert_array_equal(results['initial']['target'], results['trained']['target'])
        np.testing.assert_array_equal(results['initial']['gate'], results['trained']['gate'])
        report['evaluations'][kind] = {label:dict(
            groups=continuous_scores(r, times),
            firing=report_run(r, np.arange(len(frames)-1), metadata['neurons']['dt']))
            for label,r in results.items()}
        m = report['evaluations'][kind]
        print(kind, 'initial', json.dumps(m['initial']['groups']['all']['learned']),
              'trained', json.dumps(m['trained']['groups']['all']['learned']), flush=True)
        (OUT/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for label in ['standard', 'omitted']:
        initial_path = OUT/f'continuous-initial-{label}.npz'
        trained_path = OUT/f'continuous-trained-{label}.npz'
        assert checksum(initial_path) == report['artifacts'][str(initial_path)]
        assert checksum(trained_path) == report['artifacts'][str(trained_path)]
    report['script_sha256'] = checksum(Path(__file__))
    (OUT/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
