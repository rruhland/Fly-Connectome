"""Offline causal reference calibration; never installs a neural mechanism."""
import json
from pathlib import Path

import numpy as np

from credit_capacity import score
from fly_connectome.data import checksum
from temporal_visual import recurrent_boundaries


def sensory_history(events, decay, steps=8):
    state = np.float32(0)
    result = []
    for event in events:
        for tick in range(steps):
            state = np.float32(state*np.float32(decay))
            if tick == 0:
                state = np.float32(state + np.float32(30*event))
                result.append(float(state))
    return np.asarray(result)


def causal_window(state, events, frame_decay):
    reference = 0.
    references = np.zeros(len(state))
    gate = np.zeros(len(state), dtype=bool)
    margin = np.sqrt(frame_decay)
    for frame, value in enumerate(abs(state)):
        if frame > 0 and events[frame] != 0:
            reference = abs(state[frame-1])
        references[frame] = reference
        gate[frame] = reference > 0 and reference*margin <= value <= reference/margin
    return gate, references


def main():
    source = Path('runs/credit-capacity-v1/predictions.npz')
    manifest = json.loads(source.with_name('results.json').read_text())
    digest = checksum(source)
    assert digest == manifest['artifacts'][str(source)]
    data = np.load(source)
    config = json.loads(Path('runs/temporal-area-matched-excitation-v1/manifest.json').read_text())['neurons_config']
    decay = np.exp(-config['dt']/config['tau_sensory'])
    results = {}; histories = {}; source_hashes = {}
    for dwell in [2, 3, 4, 6]:
        path = Path(f'runs/context-audit-v1/dwell-{dwell}.npz')
        source_hashes[str(dwell)] = checksum(path)
        assert source_hashes[str(dwell)] == manifest['reconstruction'][str(dwell)]['sha256']
        original = np.load(path)
        events = original['all_target'].reshape(-1)[::8]
        state = sensory_history(events, decay)
        issue = recurrent_boundaries(original['blank_frames'], dwell=dwell)//8
        keep = original['trial'] > 0
        issue = issue[keep]
        mask = data['tempo'] == dwell
        np.testing.assert_allclose(state[issue], data['state'][mask], atol=1e-4, rtol=1e-5)
        np.testing.assert_array_equal(events[issue+1], data['y'][mask])
        gate, reference = causal_window(state, events, decay**8)
        test = data['test'][mask]
        p = np.clip(data['split-balanced'][mask][test], -1, 1)
        y = data['y'][mask][test]
        gated = p*gate[issue[test]]
        result = score(y, gated)
        result['ungated'] = score(y, p)
        result['retention'] = {name: float((gated[m]*y[m]).sum()/(p[m]*y[m]).sum())
            for name, m in [('ON', y < 0), ('OFF', y > 0)]}
        result['reconstruction_max_error'] = float(abs(state[issue]-data['state'][mask]).max())
        results[str(dwell)] = result
        histories[f'dwell{dwell}_gate'] = gate
        histories[f'dwell{dwell}_reference'] = reference
    # Continuous sensory-only stress: metadata is used for reporting, never by the selector.
    times = []; blocks = []; t = 20
    for block, interval in enumerate([2, 3, 6, 4]):
        for ordinal in range(24):
            t += interval
            times.append(t); blocks.append(block)
    events = np.zeros(t+30)
    events[times] = np.resize([-1., 1.], len(times))
    stress = {}
    for condition in ['standard', 'omitted']:
        observed = events.copy()
        if condition == 'omitted':
            observed[np.asarray(times)[60:62]] = 0
        state = sensory_history(observed, decay)
        gate, reference = causal_window(state, observed, decay**8)
        rows = []
        for i, (frame, block) in enumerate(zip(times, blocks)):
            rows.append(dict(block=block, ordinal=i%24, frame=frame,
                event=float(observed[frame]), predicted=bool(gate[frame-1]),
                reference=float(reference[frame-1]), state=float(abs(state[frame-1]))))
        quiet = observed[1:] == 0
        stress[condition] = dict(scheduled_events=rows,
            quiet_activations=int(gate[:-1][quiet].sum()), quiet_frames=int(quiet.sum()),
            quiet_activation_frames=(np.flatnonzero(gate[:-1] & quiet)+1).tolist())
        histories[f'{condition}_events'] = observed
        histories[f'{condition}_gate'] = gate
    output = Path('runs/causal-timing-v1'); output.mkdir(exist_ok=True)
    np.savez_compressed(output/'replay.npz', **histories)
    result = dict(source_sha256=digest, context_sha256=source_hashes,
        reference_coefficients=manifest['variants']['split-balanced']['coefficients'],
        tick_decay=float(decay), frame_decay=float(decay**8), held_out=results,
        sensory_only_stress=stress, script_sha256=checksum(Path(__file__)),
        replay_sha256=checksum(output/'replay.npz'))
    Path('docs/experiments/2026-09-22-causal-timing-results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    assert checksum(source) == digest
    print(json.dumps(results, indent=2))
    for condition, row in stress.items():
        print(condition, 'quiet activations', row['quiet_activations'],
              'missed event ordinals', [(r['block'], r['ordinal']) for r in row['scheduled_events']
                                      if r['event'] and not r['predicted']])


if __name__ == '__main__':
    main()
