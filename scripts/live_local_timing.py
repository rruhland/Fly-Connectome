"""Headless live test of one target's causal local prediction rule."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from causal_timing import causal_window
from continuous_timing import challenge, summarize
from controlled_visual import make_network
from credit_capacity import score
from multitempo import training_schedule
from temporal_visual import oscillation
from timed_local_prediction import TimedLocalPrediction
from timing_transfer import load_model
from fly_connectome.data import checksum
from fly_connectome.plasticity import LearningConfig
from fly_connectome.sensor import EventCamera, Retina


OUT = Path('runs/live-local-timing-v1')


def mixed_boundaries(blanks, dwells):
    lengths = np.asarray(blanks)+8*np.asarray(dwells)+1
    keep = np.ones(int(lengths.sum()), dtype=bool)
    start = 0
    for blank, dwell, length in zip(blanks, dwells, lengths):
        keep[start+int(blank):start+int(blank)+2*int(dwell)] = False
        start += int(length)
    future = np.flatnonzero(keep)
    return future[future > 0]-1


def score_run(data, issue):
    y = data['target'][issue+1]
    return dict(learned=score(y, data['gated'][issue]),
                persistence=score(y, data['target'][issue]),
                zero=score(y, np.zeros(len(issue))))


def choice(rows):
    best = None
    for eta, row in rows.items():
        m = row['validation']['learned']
        anticipation = min(m['on_anticipation'], m['off_anticipation'])
        if m['false_alarm_fraction'] <= .05:
            passing = anticipation >= .1
            key = (2 if passing else 1, -m['mse'] if passing else anticipation,
                   anticipation if passing else -m['mse'])
        else:
            key = (0, -m['false_alarm_fraction'], anticipation)
        if best is None or key > best[0]:
            best = key, eta
    return best[1]


@torch.no_grad()
def run_frames(crop, metadata, weights, frames, eta):
    net = make_network(crop, metadata, weights, predictive_kinetics='area-matched-excitation-v1')
    retina = Retina(**crop['retina']); camera = EventCamera(1, 32, 64)
    cfg = replace(LearningConfig(**metadata['learning']), eta_prediction=eta,
                  eta_reward=0., homeostasis_rate=0.)
    target = int(np.searchsorted(crop['graph'].body_ids, 82450))
    rule = TimedLocalPrediction(net, cfg, target=target,
                                sensory_mask=retina.injected,
                                sensory_gain=metadata['config']['sensory_gain'])
    incoming = ((net.post == target) & (net.pathways == 1)).nonzero().flatten()
    lookup = torch.full((net.e,), -1, dtype=torch.long); lookup[incoming] = torch.arange(len(incoming))
    history = {name: [] for name in ['raw', 'gated', 'target', 'gate', 'state', 'reference',
                                     'gain', 'delta', 'weights', 'spikes']}
    started = time.perf_counter()
    maxima = np.zeros(4)
    for image in frames:
        injection = retina.project(camera.observe(image))*metadata['config']['sensory_gain']
        for tick in range(8):
            a = net.step(injection if tick == 0 else torch.zeros_like(injection), capture_increments=True)
            rule.observe(a, torch.zeros(1))
            history['spikes'].append(a.spikes[0].numpy().copy())
            if tick == 0:
                issue = rule.last_issue
                observed = cfg.observation(a, net.config.threshold, retina.injected,
                                           metadata['config']['sensory_gain'])
                history['raw'].append(issue['raw_prediction'])
                history['gated'].append(issue['prediction'])
                history['target'].append(float(observed[0, target]))
                history['gate'].append(issue['gate'])
                history['state'].append(issue['sensory_state'])
                history['reference'].append(issue['reference'])
                confirmation = rule.last_confirmation
                d = np.zeros(len(incoming))
                if confirmation is not None:
                    positions = lookup[confirmation['edges']]
                    np.add.at(d, positions.numpy(), confirmation['delta'].numpy())
                history['delta'].append(d)
                history['gain'].append(1. if confirmation is None else confirmation['gain'])
        rule.synchronize()
        peak = np.array([float(getattr(net, name).abs().max()) for name in
                         ['voltage', 'sensory_state', 'predictive_current', 'adaptation']])
        if not np.isfinite(peak).all():
            raise ValueError('nonfinite neural state')
        maxima = np.maximum(maxima, peak)
        history['weights'].append(net.magnitudes[incoming].numpy().copy())
    assert torch.isfinite(net.magnitudes).all()
    assert 0 <= float(net.magnitudes.min()) and float(net.magnitudes.max()) <= cfg.maximum_weight
    result = {name: np.asarray(values) for name, values in history.items()}
    result['weights_trained'] = net.magnitudes.numpy().copy()
    result['incoming_edges'] = incoming.numpy()
    result['presynaptic_bodies'] = crop['graph'].body_ids[net.pre[incoming].numpy()]
    result['seconds'] = time.perf_counter()-started
    result['max_abs'] = maxima
    assert torch.equal(net.magnitudes[~torch.isin(torch.arange(net.e), incoming)],
                       weights[~torch.isin(torch.arange(net.e), incoming)])
    return result


def parity():
    torch.set_num_threads(1)
    crop, metadata, _ = load_model()
    assert metadata['config']['neural_steps'] == 8 and metadata['config']['sync_steps'] == 1
    source = Path('runs/multitempo-v1/mixed-training.npz')
    capacity = json.loads(Path('runs/credit-capacity-v1/results.json').read_text())
    assert checksum(source) == capacity['artifacts'][str(source)]
    weights = torch.from_numpy(np.load(source)['weights_trained'])
    old_path = Path('runs/continuous-timing-v1/standard.npz')
    old_manifest = json.loads(Path('docs/experiments/2026-09-22-continuous-timing-results.json').read_text())
    assert checksum(old_path) == old_manifest['artifacts'][str(old_path)]
    old = np.load(old_path)
    frames, _, _ = challenge(False)
    r = run_frames(crop, metadata, weights, frames, eta=0.)
    np.testing.assert_array_equal(r['presynaptic_bodies'], old['presynaptic_bodies'])
    for name, baseline in [('raw', 'prediction'), ('target', 'events'),
                           ('state', 'state'), ('spikes', 'spikes'), ('gate', 'gate'),
                           ('reference', 'reference')]:
        np.testing.assert_array_equal(r[name], old[baseline])
    np.testing.assert_array_equal(r['gated'], old['prediction']*old['gate'])
    np.testing.assert_array_equal(r['weights_trained'], weights)
    decay = np.exp(-metadata['neurons']['dt']/metadata['neurons']['tau_sensory'])**8
    gate, reference = causal_window(r['state'], r['target'], decay)
    np.testing.assert_array_equal(gate, r['gate'])
    np.testing.assert_array_equal(reference, r['reference'])
    OUT.mkdir(exist_ok=True)
    path = OUT/'parity.npz'
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **r)
    report = dict(source_sha256=checksum(source), old_sha256=checksum(old_path),
                  parity_sha256=checksum(path), script_sha256=checksum(Path(__file__)),
                  exact_fields=['raw', 'target', 'state', 'spikes', 'gate', 'reference',
                                'gated', 'weights_trained'], frames=len(frames), seconds=r['seconds'])
    (OUT/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Zero-learning parity exact:', report['exact_fields'], 'frames', len(frames), flush=True)


def starting_weights(crop, saved):
    weights = saved.clone()
    target = int(np.searchsorted(crop['graph'].body_ids, 82450))
    incoming = np.flatnonzero((crop['graph'].post == target) &
                              (np.asarray(crop['pathways']) == 'predictive'))
    table = json.loads(Path('docs/experiments/2026-09-21-short-term-learning-weights.json').read_text())
    np.testing.assert_array_equal(crop['graph'].body_ids[crop['graph'].pre[incoming]],
                                  [row['pre'] for row in table])
    np.testing.assert_array_equal(crop['weights'][incoming], [row['initial'] for row in table])
    weights[incoming] = crop['weights'][incoming]
    return weights


def report_run(r, issue, dt):
    rates = r['spikes'].mean(0)/dt
    return dict(scores=score_run(r, issue), frames=len(r['target']), seconds=float(r['seconds']),
                frames_per_second=float(len(r['target'])/r['seconds']),
                max_cell_rate_hz=float(rates.max()), mean_cell_rate_hz=float(rates.mean()),
                max_abs_state=r['max_abs'].tolist(),
                weights=r['weights_trained'][r['incoming_edges']].tolist())


def train():
    torch.set_num_threads(1)
    crop, metadata, _ = load_model()
    source = Path('runs/multitempo-v1/mixed-training.npz')
    capacity = json.loads(Path('runs/credit-capacity-v1/results.json').read_text())
    assert checksum(source) == capacity['artifacts'][str(source)]
    saved = torch.from_numpy(np.load(source)['weights_trained'])
    start = starting_weights(crop, saved)
    train_frames, train_blanks, train_dwells = training_schedule(42, 9170)
    validation_frames, validation_blanks, validation_dwells = training_schedule(9, 9171)
    train_issue = mixed_boundaries(train_blanks, train_dwells)
    validation_issue = mixed_boundaries(validation_blanks, validation_dwells)
    OUT.mkdir(exist_ok=True)
    report = dict(source_sha256=checksum(source), parity_sha256=checksum(OUT/'parity.npz'),
        training=dict(seed=9170, blanks=train_blanks.tolist(), dwells=train_dwells.tolist()),
        validation=dict(seed=9171, blanks=validation_blanks.tolist(), dwells=validation_dwells.tolist()),
        initial_target_weights=start.numpy()[np.load(OUT/'parity.npz')['incoming_edges']].tolist(),
        candidates={}, artifacts={})
    for eta in [.1, 1.]:
        key = str(eta)
        train_path = OUT/f'train-eta-{key}.npz'
        if train_path.exists():
            learned = np.load(train_path)
        else:
            learned = run_frames(crop, metadata, start, train_frames, eta=eta)
            np.savez_compressed(train_path, **learned)
            print('Trained eta', key, 'frames', len(train_frames), flush=True)
        validation_path = OUT/f'validation-eta-{key}.npz'
        if validation_path.exists():
            held = np.load(validation_path)
        else:
            held = run_frames(crop, metadata, torch.from_numpy(learned['weights_trained']),
                              validation_frames, eta=0.)
            np.savez_compressed(validation_path, **held)
            print('Validated eta', key, flush=True)
        report['artifacts'][str(train_path)] = checksum(train_path)
        report['artifacts'][str(validation_path)] = checksum(validation_path)
        report['candidates'][key] = dict(training=report_run(learned, train_issue, metadata['neurons']['dt']),
                                        validation=report_run(held, validation_issue, metadata['neurons']['dt']))
        print('eta', key, 'validation', json.dumps(report['candidates'][key]['validation']['scores']['learned']), flush=True)
        (OUT/'train.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    selection = {key:dict(validation=value['validation']['scores']) for key,value in report['candidates'].items()}
    report['selected_eta'] = choice(selection)
    report['script_sha256'] = checksum(Path(__file__))
    (OUT/'train.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Selected eta', report['selected_eta'], 'using validation only', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['parity', 'train'])
    args = parser.parse_args()
    if args.stage == 'parity':
        parity()
    elif args.stage == 'train':
        train()
