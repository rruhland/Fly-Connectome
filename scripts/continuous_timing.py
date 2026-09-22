"""Fixed offline readout and causal selector on continuous neural histories."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from causal_timing import causal_window, sensory_history
from controlled_visual import make_network
from credit_capacity import design
from frame_prediction import FramePrediction
from local_information import collect
from temporal_visual import oscillation, recurrent_boundaries
from timing_transfer import load_model
from fly_connectome.data import checksum
from fly_connectome.plasticity import LearningConfig
from fly_connectome.sensor import EventCamera, Retina


def challenge(omitted):
    times = 20+np.cumsum(np.repeat([2, 3, 6, 4], 24))
    frames = torch.zeros(int(times[-1])+31, 1, 32, 64, dtype=torch.bool)
    toggles = np.zeros(len(frames), dtype=int)
    toggles[np.delete(times, [60, 61]) if omitted else times] = 1
    position = toggles.cumsum()%2
    frames[torch.arange(len(frames)), 0, 30, torch.from_numpy(38+position)] = True
    return frames, times, np.repeat(np.arange(4), 24)


def summarize(y, p):
    result = dict(samples=len(y), mse=float(np.mean((y-p)**2)) if len(y) else None)
    for name, mask in [('on', y < 0), ('off', y > 0), ('quiet', y == 0)]:
        result[name+'_count'] = int(mask.sum())
        if name == 'quiet':
            result['quiet_alarm_fraction'] = float((abs(p[mask]) >= .1).mean()) if mask.any() else None
        else:
            result[name+'_anticipation'] = float((p[mask]*y[mask]).mean()) if mask.any() else None
    return result


@torch.no_grad()
def record(crop, metadata, weights, frames):
    net = make_network(crop, metadata, weights, predictive_kinetics='area-matched-excitation-v1')
    retina = Retina(**crop['retina']); camera = EventCamera(1, 32, 64)
    cfg = replace(LearningConfig(**metadata['learning']), eta_prediction=0., eta_reward=0., homeostasis_rate=0.)
    rule = FramePrediction(net, cfg, sensory_mask=retina.injected, sensory_gain=metadata['config']['sensory_gain'])
    target = int(np.searchsorted(crop['graph'].body_ids, 82450))
    incoming = ((net.post == target) & (net.pathways == 1)).nonzero().flatten()
    lookup = torch.full((net.e,), -1, dtype=torch.long); lookup[incoming] = torch.arange(len(incoming))
    eligibility = []; state = []; events = []; predicted = []; spikes = []
    for image in frames:
        injection = retina.project(camera.observe(image))*metadata['config']['sensory_gain']
        for tick in range(8):
            a = net.step(injection if tick == 0 else torch.zeros_like(injection), capture_increments=True)
            observed = cfg.observation(a, net.config.threshold, retina.injected, metadata['config']['sensory_gain'])
            rule.observe(a, torch.zeros(1))
            spikes.append(a.spikes[0].numpy().copy())
            if tick == 0:
                e = torch.zeros(len(incoming))
                positions = lookup[rule.keys]; keep = positions >= 0
                e[positions[keep]] = rule.values[keep]
                eligibility.append(e.numpy().copy())
                state.append(float(net.sensory_state[0, target]))
                events.append(float(observed[0, target]))
                predicted.append(float(cfg.encode(a.predicted[0, target], net.config.threshold)))
        rule.synchronize()
    assert torch.equal(net.magnitudes, weights)
    return dict(eligibility=np.asarray(eligibility), state=np.asarray(state), events=np.asarray(events),
        prediction=np.asarray(predicted), spikes=np.asarray(spikes),
        presynaptic_bodies=crop['graph'].body_ids[net.pre[incoming].numpy()])


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    crop, metadata, _ = load_model()
    weight_path = Path('runs/multitempo-v1/mixed-training.npz')
    capacity_path = Path('runs/credit-capacity-v1/results.json')
    manifest = json.loads(capacity_path.read_text())
    digest = checksum(weight_path)
    assert digest == manifest['artifacts'][str(weight_path)]
    weights = torch.from_numpy(np.load(weight_path)['weights_trained'])
    coefficients = np.asarray(manifest['variants']['split-balanced']['coefficients'])
    decay = np.exp(-metadata['neurons']['dt']/metadata['neurons']['tau_sensory'])
    assert metadata['config']['sensory_gain'] == 30
    # Independent collector control, including warmup and frame-boundary timing.
    blanks = np.array([12, 16])
    old = collect(crop, metadata, weights, blanks, dwell=3)
    control = record(crop, metadata, weights, torch.cat([oscillation(int(b)) for b in blanks]))
    issue = recurrent_boundaries(blanks, dwell=3)//8
    np.testing.assert_array_equal(control['presynaptic_bodies'], manifest['presynaptic_bodies'])
    names = old['feature_names'].tolist()
    columns = [names.index(f'eligibility_{body}') for body in control['presynaptic_bodies']]
    np.testing.assert_array_equal(control['eligibility'][issue], old['x'][:, columns])
    np.testing.assert_array_equal(control['state'][issue], old['x'][:, 3])
    np.testing.assert_array_equal(control['events'][issue+1], old['y'])
    np.testing.assert_array_equal(control['prediction'][issue], old['prediction'])
    np.testing.assert_array_equal(control['spikes'], old['all_spikes'])
    print('Collector control matches existing collector exactly', flush=True)
    output = Path('runs/continuous-timing-v1'); output.mkdir(exist_ok=False)
    results = {}; recorded = {}; hashes = {}
    for condition in ['standard', 'omitted']:
        frames, times, blocks = challenge(condition == 'omitted')
        data = record(crop, metadata, weights, frames)
        np.testing.assert_array_equal(data['presynaptic_bodies'], manifest['presynaptic_bodies'])
        expected = -np.diff(np.r_[0, frames[:, 0, 30, 39].numpy().astype(int)])
        np.testing.assert_array_equal(data['events'], expected)
        np.testing.assert_array_equal(sensory_history(data['events'], decay), data['state'])
        x = data['eligibility']
        assert all((v >= 0).all() or (v <= 0).all() for v in x.T)
        amplitude = np.clip(design(x, data['state'], True)@coefficients, -1, 1)
        gate, reference = causal_window(data['state'], data['events'], decay**8)
        gated = amplitude*gate
        y = data['events'][1:]; target_frames = np.arange(1, len(frames))
        groups = {'whole': np.ones(len(y), dtype=bool),
                  'initial_quiet': target_frames < times[0], 'trailing_quiet': target_frames > times[-1]}
        for block, dwell in enumerate([2, 3, 6, 4]):
            start = block*24
            end = times[start+24] if block < 3 else times[-1]+1
            within = (target_frames >= times[start]) & (target_frames < end)
            recovery = (target_frames >= times[60]) & (target_frames < times[64]) if block == 2 else np.zeros(len(y), bool)
            groups[f'dwell{dwell}_all'] = within
            groups[f'dwell{dwell}_adaptation'] = within & (target_frames < times[start+4])
            groups[f'dwell{dwell}_steady'] = within & (target_frames >= times[start+4]) & ~recovery
            if block == 2:
                groups['omission_window'] = within & recovery
        report = {name: dict(ungated=summarize(y[mask], amplitude[:-1][mask]),
                            gated=summarize(y[mask], gated[:-1][mask])) for name, mask in groups.items()}
        rows = [dict(block=int(block), ordinal=i%24, frame=int(frame), target=float(data['events'][frame]),
                     amplitude=float(amplitude[frame-1]), gate=bool(gate[frame-1]),
                     gated=float(gated[frame-1])) for i, (frame, block) in enumerate(zip(times, blocks))]
        results[condition] = dict(groups=report, scheduled_events=rows,
            quiet_alarm_frames=(np.flatnonzero((y == 0) & (abs(gated[:-1]) >= .1))+1).tolist(),
            active_eligibility_columns=np.flatnonzero(np.any(x != 0, axis=0)).tolist())
        data.update(amplitude=amplitude, gate=gate, reference=reference, gated=gated,
                    scheduled_frames=times, scheduled_blocks=blocks)
        path = output/f'{condition}.npz'; np.savez_compressed(path, **data)
        hashes[str(path)] = checksum(path); recorded[condition] = data
        print(condition, json.dumps(report['whole']), flush=True)
    first_omission = int(times[60])
    for key in ['eligibility', 'state', 'events', 'prediction', 'amplitude', 'gate', 'reference', 'gated']:
        np.testing.assert_array_equal(recorded['standard'][key][:first_omission], recorded['omitted'][key][:first_omission])
    np.testing.assert_array_equal(recorded['standard']['spikes'][:8*first_omission],
                                  recorded['omitted']['spikes'][:8*first_omission])
    assert checksum(weight_path) == digest
    report = dict(seconds=time.perf_counter()-started, collector_control_exact=True, omission_prefix_exact=True,
        frozen_weights_sha256=digest, capacity_manifest_sha256=checksum(capacity_path),
        graph_identity=crop['graph'].identity(), coefficients=coefficients.tolist(),
        script_sha256=checksum(Path(__file__)), artifacts=hashes, results=results)
    Path('docs/experiments/2026-09-22-continuous-timing-results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
