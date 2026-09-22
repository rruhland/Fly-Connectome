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
from timed_local_prediction import TimedLocalPrediction
from timing_transfer import load_model
from fly_connectome.data import checksum
from fly_connectome.plasticity import LearningConfig
from fly_connectome.sensor import EventCamera, Retina


OUT = Path('runs/live-local-timing-v1')


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
        history['weights'].append(net.magnitudes[incoming].numpy().copy())
    assert torch.isfinite(net.magnitudes).all()
    assert 0 <= float(net.magnitudes.min()) and float(net.magnitudes.max()) <= cfg.maximum_weight
    result = {name: np.asarray(values) for name, values in history.items()}
    result['weights_trained'] = net.magnitudes.numpy().copy()
    result['incoming_edges'] = incoming.numpy()
    result['presynaptic_bodies'] = crop['graph'].body_ids[net.pre[incoming].numpy()]
    result['seconds'] = time.perf_counter()-started
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['parity'])
    args = parser.parse_args()
    if args.stage == 'parity':
        parity()
