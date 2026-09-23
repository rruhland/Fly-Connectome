"""Read-only location-transfer probe for motion along the measured visual graph."""
import argparse
import json
from pathlib import Path

import torch

from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina


SOURCE = Path('checkpoints/event-v1-combined-rate-initial.pt')
OUT = Path('runs/motion-stage-audit-v1')
LABELS = ('L1', 'L2', 'L3', 'L4', 'Mi1', 'Tm3', 'Mi4', 'Mi9',
          'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T4a', 'T4b', 'T4c', 'T4d',
          'T5a', 'T5b', 'T5c', 'T5d', 'LPi')


def frames_for_condition(center, direction, polarity, *, kind='dot'):
    """Matched 13-position trajectories, two blank lead and three blank tail frames."""
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    background = polarity == 'off'
    frames = []
    for frame in range(18):
        if 2 <= frame < 15:
            dot_x = center + direction*(frame-8)
            dot = ((x-dot_x).square()+(y-16).square() <= 1.5**2 if kind == 'dot'
                   else ((x-dot_x).abs() <= 1) & ((y-16).abs() <= 6))
            pixels = dot if polarity == 'on' else ~dot
        else:
            pixels = torch.full((32, 64), background)
        frames.append(pixels.unsqueeze(0))
    return frames


def local_columns(retina, center):
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    path = (x >= center-8) & (x <= center+8) & (y >= 13) & (y <= 19)
    bins = retina.pixel_bins[path.flatten()].unique()
    return torch.isin(retina.neuron_columns, bins)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stimulus', choices=('dot', 'bar'), default='dot')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    m = payload['metadata']
    retina = Retina(**m['retina'])
    graph = Graph(**m['graph'], gain=m['gain'])
    net = MultiContextEfficacyNetwork(graph, m['delays'], m['pathways'],
        config=NeuronConfig(**m['neurons']), cell_types=m['retina']['cell_types'],
        target_mask=retina.injected)
    original = payload['state']['network']['magnitudes']
    net.set_weights(original.clamp(0, m['learning']['maximum_weight']))
    zero = torch.zeros_like(net.voltage)
    for _ in range(m['config']['warmup_steps']):
        net.step(zero)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    warm_tick = net.step_index
    camera = EventCamera(1, 32, 64)
    types = m['retina']['cell_types']
    masks = {label: torch.tensor([t.startswith('LPi') if label == 'LPi' else t == label
                                  for t in types]) for label in LABELS}
    regions = {center: local_columns(retina, center) for center in (18, 46)}

    def probe(frames, polarity):
        for name, state in warm.items():
            getattr(net, name).copy_(state)
        net.step_index = warm_tick
        camera.previous.fill_(polarity == 'off')
        spikes = torch.zeros(net.n, dtype=torch.int32)
        voltage = torch.zeros(net.n)
        pixel_events = neural_inputs = 0
        for frame, image in enumerate(frames):
            events = camera.observe(image)
            injection = retina.project(events)*m['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
                neural_inputs += int((injection != 0).sum())
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                if 3 <= frame < 15:
                    spikes.add_(activity.spikes[0].int())
                    voltage.add_(net.voltage[0])
        return dict(spikes=spikes, mean_voltage=voltage/96,
                    transit_pixel_events=pixel_events, transit_neural_inputs=neural_inputs)

    baselines = {
        polarity: probe([torch.full((1, 32, 64), polarity == 'off')]*18, polarity)
        for polarity in ('on', 'off')
    }
    responses = {}
    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
                  warmup_ticks=m['config']['warmup_steps'], ticks_per_frame=8,
                  frames_per_condition=18, source='initial', stimulus=args.stimulus,
                  groups={})
    for center in (18, 46):
        for polarity in ('on', 'off'):
            baseline = baselines[polarity]
            for direction in (1, -1):
                key = f'{center}-{polarity}-{direction:+d}'
                response = probe(frames_for_condition(center, direction, polarity,
                                                      kind=args.stimulus), polarity)
                responses[key] = response
                row = dict(transit_pixel_events=response['transit_pixel_events'],
                           transit_neural_inputs=response['transit_neural_inputs'],
                           groups={})
                for label, type_mask in masks.items():
                    if not type_mask.any():
                        continue
                    for region, mask in (('global', type_mask),
                                         ('local', type_mask & regions[center])):
                        if not mask.any():
                            continue
                        spikes = response['spikes'][mask]
                        blank_spikes = baseline['spikes'][mask]
                        row['groups'][f'{label}:{region}'] = dict(
                            neurons=int(mask.sum()),
                            spikes=int(spikes.sum()), blank_spikes=int(blank_spikes.sum()),
                            excess_spikes=int((spikes-blank_spikes).sum()),
                            changed_spike_counts=int((spikes != blank_spikes).sum()),
                            mean_voltage_change=float((response['mean_voltage'][mask]
                                - baseline['mean_voltage'][mask]).mean()))
                report['groups'][key] = row
            plus = report['groups'][f'{center}-{polarity}-+1']
            minus = report['groups'][f'{center}-{polarity}--1']
            if plus['transit_pixel_events'] != minus['transit_pixel_events']:
                raise AssertionError('opposite directions had different pixel event totals')
    report['direction_contrasts'] = {}
    for polarity in ('on', 'off'):
        for label in LABELS:
            for region in ('global', 'local'):
                key = f'{label}:{region}'
                values = []
                for center in (18, 46):
                    right = report['groups'][f'{center}-{polarity}-+1']['groups']
                    left = report['groups'][f'{center}-{polarity}--1']['groups']
                    if key in right and key in left:
                        values.append(right[key]['excess_spikes']-left[key]['excess_spikes'])
                if len(values) == 2:
                    report['direction_contrasts'][f'{polarity}:{key}'] = dict(
                        right_minus_left=values,
                        sign_consistent=values[0]*values[1] > 0)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.mkdir(parents=True, exist_ok=True)
    raw_name = ('per-neuron-responses.pt' if args.stimulus == 'dot'
                else 'bar-per-neuron-responses.pt')
    torch.save(dict(responses=responses, baselines=baselines,
                    retina_columns=retina.neuron_columns, cell_types=types),
               OUT/raw_name)
    path = OUT/('initial-results.json' if args.stimulus == 'dot' else 'bar-results.json')
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path),
        t4_t5_contrasts={key: value for key, value in report['direction_contrasts'].items()
                         if ':T4' in key or ':T5' in key})), flush=True)


if __name__ == '__main__':
    main()
