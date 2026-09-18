"""Frozen preregistered operating-point probes; no Pong rollout or weight updates."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path

import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import Network
from fly_connectome.sensor import Retina, EventCamera
from fly_connectome.training import load_checkpoint


def select_profile(rows, cell_types):
    """Conservative modulation screen, then preregistered total-current cost."""
    counts = Counter(cell_types)
    candidates = []
    for row in rows:
        p = row['populations']
        responsive = (all(p[t]['stimulus_counts'][0] > p[t]['stimulus_counts'][2] for t in ('Mi1', 'Tm3'))
                      and all(p[t]['stimulus_counts'][1] > p[t]['stimulus_counts'][2] for t in ('Tm1', 'Tm2')))
        if row['stable'] and responsive:
            cost = sum(counts[t] * values['rest_current'] for t, values in row['neurons_config']['class_parameters'].items())
            candidates.append((cost, row))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/resting-calibration-v1.json')
    parser.add_argument('--output', default='runs/resting-calibration-v1.json')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    spec = json.loads(Path(args.config).read_text())
    source = load_checkpoint(spec['source_checkpoint'], evaluation=True, seeds=[spec['seed']])
    assert source.network.graph.gain == spec['gain']
    types = source.retina.spec['cell_types']
    labels = ['L1', 'L2', 'L3', 'L4', 'Mi1', 'Tm3', 'Tm1', 'Tm2', 'Mi9', 'T4', 'T5', 'LPi', 'LC10']
    masks = {label: torch.tensor([t == label or (label in ('T4','T5','LPi','LC10') and t.startswith(label)) for t in types]) for label in labels}
    retina = Retina(**dict(source.retina.spec, injection=dict.fromkeys(['L1','L2','L3'], 'contrast')))
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for lamina, medulla in itertools.product(spec['lamina_rest_currents'], spec['medulla_rest_currents']):
        parameters = {t: dict(p, rest_current=lamina if t in ('L1','L2','L3') else medulla)
                      for t, p in spec['class_timescales'].items()}
        config = replace(source.network.config, class_parameters=parameters, tau_sensory=spec['tau_sensory'])
        net = Network(source.network.graph, source.network.delays.tolist(),
                      [('feedforward','predictive','behavioral')[p] for p in source.network.pathways.tolist()],
                      config=config, cell_types=types, batch=3)
        camera = EventCamera(3, 32, 64)
        camera.previous[1].fill_(True)
        baseline = torch.zeros(3, net.n, dtype=torch.int64)
        response = torch.zeros_like(baseline)
        recovery = torch.zeros_like(baseline)
        peak, stable = 0, True
        for tick in range(spec['warmup_ticks'] + spec['probe_ticks']):
            frame = torch.zeros(3,32,64,dtype=torch.bool)
            frame[1].fill_(True)
            elapsed = tick - spec['warmup_ticks']
            if 0 <= elapsed < 200:
                edge = min(64, int(elapsed * .32))
                frame[0, :, :edge] = True
                frame[1, :, :edge] = False
            # Third private environment is the matched no-event baseline.
            activity = net.step(retina.project(camera.observe(frame)) * source.config.sensory_gain)
            peak = max(peak, int(activity.spikes.sum(1).max()))
            if not torch.isfinite(net.voltage).all() or peak > net.n * .2:
                stable = False
                break
            if spec['warmup_ticks'] // 2 <= tick < spec['warmup_ticks']:
                baseline += activity.spikes
            if 0 <= elapsed < 200:
                response += activity.spikes
            if elapsed >= 200:
                recovery += activity.spikes
        row = dict(lamina_current=lamina, medulla_current=medulla, stable=stable,
                   peak_spikes_per_tick=peak, ticks=tick+1, neurons=net.n, neurons_config=asdict(config),
                   populations={label: dict(baseline_counts=baseline[:,mask].sum(1).tolist(),
                       stimulus_counts=response[:,mask].sum(1).tolist(), recovery_counts=recovery[:,mask].sum(1).tolist())
                       for label, mask in masks.items()})
        rows.append(row)
        output.write_text(json.dumps(dict(config=spec, config_sha256=checksum(args.config),
            checkpoint_sha256=checksum(spec['source_checkpoint']), conditions=['on','off','no_event'],
            results=rows), indent=2) + '\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('neurons_config',)}), flush=True)
    chosen = select_profile(rows, types)
    if chosen is not None:
        profile = dict(id='resting-v1-provisional', status='probe-selected approximation; M1A learning unverified',
            neurons=chosen['neurons_config'], injection=dict.fromkeys(['L1','L2','L3'], 'contrast'),
            warmup_steps=spec['warmup_ticks'], calibration_config_sha256=checksum(args.config),
            calibration_checkpoint_sha256=checksum(spec['source_checkpoint']))
        output.with_name('resting-v1-provisional.json').write_text(json.dumps(profile, indent=2) + '\n')
    else:
        print('No candidate passed the response screen; no profile selected.', flush=True)


if __name__ == '__main__':
    main()
