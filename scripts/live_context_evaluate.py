"""Untouched tempo and continuous-stream evaluation of local context efficacy."""
import json
from pathlib import Path

import numpy as np
import torch

from continuous_timing import challenge
from live_context_efficacy import OUT, run_frames
from live_local_evaluate import continuous_scores, save
from live_local_timing import report_run, starting_weights
from temporal_visual import oscillation, recurrent_boundaries
from timing_transfer import load_model
from fly_connectome.data import checksum


def main():
    torch.set_num_threads(1)
    crop, metadata, _ = load_model()
    selection_path = OUT/'train.json'
    selected = json.loads(selection_path.read_text())
    source = Path('runs/multitempo-v1/mixed-training.npz')
    if checksum(source) != selected['source_sha256']:
        raise ValueError('source weights changed')
    initial = starting_weights(crop, torch.from_numpy(np.load(source)['weights_trained']))
    train_path = OUT/f"train-eta-{selected['selected_eta']}.npz"
    if checksum(train_path) != selected['artifacts'][str(train_path)]:
        raise ValueError('selected training artifact changed')
    trained = np.load(train_path)
    edge = trained['incoming_edges']
    outside = np.ones(len(initial), bool); outside[edge] = False
    np.testing.assert_array_equal(trained['weights_trained'][outside], initial.numpy()[outside])
    previous = np.concatenate((np.repeat(initial.numpy()[edge][None, None, :], 2, axis=1),
                               trained['weights'][:-1]), axis=0)
    proposed = np.clip(previous+trained['delta'], 0, 10)
    np.testing.assert_allclose(proposed, trained['weights'], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(trained['weights'][-1], trained['components_trained'])
    report = dict(training_sha256=checksum(train_path), selection_sha256=checksum(selection_path),
        source_sha256=checksum(source), initial_target_weights=initial.numpy()[edge].tolist(),
        trained_components=trained['components_trained'].tolist(),
        audit_max_abs=float(abs(proposed-trained['weights']).max()),
        evaluations={}, artifacts={})
    conditions = [('initial', None), ('trained', torch.from_numpy(trained['components_trained']))]
    blanks = np.random.default_rng(9172).integers(12, 37, size=8)
    report['heldout_blanks'] = blanks.tolist()
    shared = json.loads(Path('runs/live-local-timing-v1/evaluation.json').read_text())
    for dwell in [2, 3, 4, 6]:
        frames = torch.cat([oscillation(int(blank), dwell=dwell) for blank in blanks])
        issue = recurrent_boundaries(blanks, dwell=dwell)//8
        results = {}
        for label, components in conditions:
            result = run_frames(crop, metadata, initial, frames, eta=0., components=components)
            results[label] = result
            path = OUT/f'heldout-{label}-dwell{dwell}.npz'
            report['artifacts'][str(path)] = save(path, result)
            if label == 'initial':
                previous_path = Path(f'runs/live-local-timing-v1/heldout-initial-dwell{dwell}.npz')
                assert checksum(previous_path) == shared['artifacts'][str(previous_path)]
                old = np.load(previous_path)
                for field in ('raw', 'gated', 'target', 'spikes', 'gate'):
                    np.testing.assert_array_equal(result[field], old[field])
        np.testing.assert_array_equal(results['initial']['target'], results['trained']['target'])
        np.testing.assert_array_equal(results['initial']['gate'], results['trained']['gate'])
        report['evaluations'][f'dwell{dwell}'] = {label:report_run(result, issue, metadata['neurons']['dt'])
                                                  for label,result in results.items()}
        m = report['evaluations'][f'dwell{dwell}']
        print('dwell', dwell, 'initial', json.dumps(m['initial']['scores']['learned']),
              'trained', json.dumps(m['trained']['scores']['learned']), flush=True)
        (OUT/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for kind in ('standard', 'omitted'):
        frames, times, _ = challenge(kind == 'omitted')
        results = {}
        for label, components in conditions:
            result = run_frames(crop, metadata, initial, frames, eta=0., components=components)
            results[label] = result
            path = OUT/f'continuous-{label}-{kind}.npz'
            report['artifacts'][str(path)] = save(path, result)
        np.testing.assert_array_equal(results['initial']['target'], results['trained']['target'])
        np.testing.assert_array_equal(results['initial']['gate'], results['trained']['gate'])
        report['evaluations'][kind] = {label:dict(
            groups=continuous_scores(result, times),
            firing=report_run(result, np.arange(len(frames)-1), metadata['neurons']['dt']))
            for label,result in results.items()}
        m = report['evaluations'][kind]
        print(kind, 'initial', json.dumps(m['initial']['groups']['all']['learned']),
              'trained', json.dumps(m['trained']['groups']['all']['learned']), flush=True)
        (OUT/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    report['script_sha256'] = checksum(Path(__file__))
    (OUT/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
