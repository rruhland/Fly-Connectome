"""Live coupled test of local context efficacy on the measured L3 motif."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from context_efficacy import ContextEfficacyNetwork, ContextTimedPrediction
from continuous_timing import challenge
from live_local_timing import choice, mixed_boundaries, report_run, starting_weights
from timing_transfer import load_model
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.plasticity import LearningConfig
from fly_connectome.sensor import EventCamera, Retina


OUT = Path('runs/live-context-efficacy-v1')


def make_network(crop, metadata, weights, components=None):
    target = int(np.searchsorted(crop['graph'].body_ids, 82450))
    net = ContextEfficacyNetwork(crop['graph'], crop['delays'], crop['pathways'],
        config=NeuronConfig(**metadata['neurons']), cell_types=crop['retina']['cell_types'],
        target=target)
    net.set_weights(weights, components)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(torch.zeros_like(net.voltage))
    return net


@torch.no_grad()
def run_frames(crop, metadata, weights, frames, eta, components=None):
    net = make_network(crop, metadata, weights, components)
    retina = Retina(**crop['retina'])
    camera = EventCamera(1, 32, 64)
    cfg = replace(LearningConfig(**metadata['learning']), eta_prediction=eta,
                  eta_reward=0., homeostasis_rate=0.)
    rule = ContextTimedPrediction(net, cfg, target=net.target,
        sensory_mask=retina.injected, sensory_gain=metadata['config']['sensory_gain'])
    incoming = net.incoming
    lookup = net.incoming_lookup
    history = {name: [] for name in ['raw', 'gated', 'target', 'gate', 'state', 'reference',
        'gain', 'delta', 'weights', 'spikes', 'context', 'eligibility']}
    maxima = np.zeros(6)
    started = time.perf_counter()
    for image in frames:
        injection = retina.project(camera.observe(image))*metadata['config']['sensory_gain']
        for tick in range(8):
            activity = net.step(injection if tick == 0 else torch.zeros_like(injection),
                                capture_increments=True)
            rule.observe(activity, torch.zeros(1))
            history['spikes'].append(activity.spikes[0].numpy().copy())
            if tick == 0:
                issue = rule.last_issue
                observed = cfg.observation(activity, net.config.threshold, retina.injected,
                                           metadata['config']['sensory_gain'])
                history['raw'].append(issue['raw_prediction'])
                history['gated'].append(issue['prediction'])
                history['target'].append(float(observed[0, net.target]))
                history['gate'].append(issue['gate'])
                history['state'].append(issue['sensory_state'])
                history['reference'].append(issue['reference'])
                history['context'].append(issue['context'])
                eligible = np.zeros(len(incoming))
                np.add.at(eligible, lookup[issue['edges']].numpy(), issue['eligibility'].numpy())
                history['eligibility'].append(eligible)
                confirmation = rule.last_confirmation
                delta = np.zeros((2, len(incoming)))
                if confirmation is not None:
                    positions = lookup[confirmation['edges']]
                    np.add.at(delta[confirmation['context']], positions.numpy(),
                              confirmation['delta'].numpy())
                history['delta'].append(delta)
                history['gain'].append(1. if confirmation is None else confirmation['gain'])
        rule.synchronize()
        peak = np.array([float(getattr(net, name).abs().max()) for name in
                         ['voltage', 'sensory_state', 'predictive_current', 'adaptation',
                          'context_exc', 'context_inh']])
        if not np.isfinite(peak).all():
            raise ValueError('nonfinite neural state')
        maxima = np.maximum(maxima, peak)
        history['weights'].append(net.components.numpy().copy())
    if not torch.isfinite(net.components).all() or not (0 <= float(net.components.min()) <=
                                                          float(net.components.max()) <= cfg.maximum_weight):
        raise ValueError('context magnitude outside fixed bounds')
    outside = torch.ones(net.e, dtype=torch.bool); outside[incoming] = False
    torch.testing.assert_close(net.magnitudes[outside], weights[outside], rtol=0, atol=0)
    result = {name: np.asarray(values) for name, values in history.items()}
    result['weights_trained'] = net.magnitudes.numpy().copy()
    result['components_trained'] = net.components.numpy().copy()
    result['incoming_edges'] = incoming.numpy()
    result['presynaptic_bodies'] = crop['graph'].body_ids[net.pre[incoming].numpy()]
    result['seconds'] = time.perf_counter()-started
    result['max_abs'] = maxima
    return result


def parity():
    torch.set_num_threads(1)
    crop, metadata, _ = load_model()
    source = Path('runs/multitempo-v1/mixed-training.npz')
    capacity = json.loads(Path('runs/credit-capacity-v1/results.json').read_text())
    assert checksum(source) == capacity['artifacts'][str(source)]
    weights = torch.from_numpy(np.load(source)['weights_trained'])
    old_path = Path('runs/live-local-timing-v1/parity.npz')
    old_report = json.loads(Path('runs/live-local-timing-v1/parity.json').read_text())
    assert checksum(old_path) == old_report['parity_sha256']
    old = np.load(old_path)
    frames, _, _ = challenge(False)
    current = run_frames(crop, metadata, weights, frames, eta=0.)
    for name in ['raw', 'gated', 'target', 'gate', 'state', 'reference', 'gain',
                 'spikes', 'incoming_edges', 'presynaptic_bodies', 'weights_trained']:
        np.testing.assert_array_equal(current[name], old[name], err_msg=name)
    np.testing.assert_array_equal(current['components_trained'],
                                  np.repeat(weights[current['incoming_edges']][None, :], 2, axis=0))
    OUT.mkdir(exist_ok=True)
    path = OUT/'parity.npz'
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **current)
    report = dict(source_sha256=checksum(source), old_sha256=checksum(old_path),
                  parity_sha256=checksum(path), frames=len(frames), seconds=current['seconds'],
                  exact_fields=['raw', 'gated', 'target', 'gate', 'state', 'reference',
                                'gain', 'spikes', 'incoming_edges', 'presynaptic_bodies',
                                'weights_trained'])
    (OUT/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Equal-component parity exact:', report['exact_fields'], 'frames', len(frames), flush=True)


def train():
    from multitempo import training_schedule
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
            held = run_frames(crop, metadata, start, validation_frames, eta=0.,
                              components=torch.from_numpy(learned['components_trained']))
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
    else:
        train()
