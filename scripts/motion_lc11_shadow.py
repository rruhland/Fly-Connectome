"""Opt-in partial LC11 readout of measured T2/T2a/T3 contacts."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina
from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from motion_lc11_wiring import WEIGHTS, collect_target_edges
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS
from motion_t3_target import static_dot


OUT = Path('docs/experiments/2026-09-23-lc11-shadow-results.json')
RAW = Path('runs/motion-lc11-shadow-v1/per-tick.npz')
ADAPTIVE_OUT = Path('docs/experiments/2026-09-23-lc11-adaptive-results.json')
ADAPTIVE_RAW = Path('runs/motion-lc11-adaptive-v1/per-tick.npz')
FRACTIONS = (.05, .10, .25)
VOLTAGE_SCALE = .003
BASELINE_TAU = .250
TRANSIT = slice(3*8, 15*8)
SOURCE_TYPES = ('T2', 'T2a', 'T3')


class ShadowLC11:
    """Only measured incoming object-pathway contacts, with one-tick delay."""

    def __init__(self, source_edge, target_edge, magnitudes, n_sources,
                 n_targets, config):
        self.source_edge = torch.as_tensor(source_edge, dtype=torch.long)
        self.target_edge = torch.as_tensor(target_edge, dtype=torch.long)
        self.magnitudes = torch.as_tensor(magnitudes, dtype=torch.float32)
        self.n_sources, self.n_targets = n_sources, n_targets
        self.current_decay = math.exp(-config.dt/config.tau_current)
        self.membrane_decay = math.exp(-config.dt/config.tau_membrane)
        self.adaptation_decay = math.exp(-config.dt/config.tau_adaptation)
        self.config = config
        self.reset()

    def reset(self):
        self.current = torch.zeros(self.n_targets)
        self.voltage = torch.zeros(self.n_targets)
        self.adaptation = torch.zeros(self.n_targets)
        self.refractory = torch.zeros(self.n_targets, dtype=torch.long)
        self.previous_release = torch.zeros(self.n_sources)

    @torch.no_grad()
    def step(self, release):
        if release.shape != (self.n_sources,):
            raise ValueError('one release value per measured presynaptic cell')
        impulse = torch.zeros(self.n_targets)
        impulse.index_add_(0, self.target_edge,
                           self.magnitudes*self.previous_release[self.source_edge])
        self.current.mul_(self.current_decay).add_(impulse)
        self.adaptation.mul_(self.adaptation_decay)
        eligible = self.refractory == 0
        self.refractory.sub_(1).clamp_(min=0)
        self.voltage.mul_(self.membrane_decay).add_(
            self.current, alpha=1-self.membrane_decay)
        self.voltage.masked_fill_(~eligible, 0.)
        spikes = eligible & (self.voltage >= self.config.threshold + self.adaptation)
        self.voltage.masked_fill_(spikes, 0.)
        self.refractory.masked_fill_(spikes, self.config.refractory_steps)
        self.adaptation.add_(spikes*self.config.adaptation_jump)
        self.previous_release.copy_(release)
        return spikes


def release_from_voltage(voltage, rest, fraction):
    return ((voltage-rest).clamp(min=0) * (fraction/VOLTAGE_SCALE)).clamp(
        max=fraction)


def adaptive_release(voltage, baseline, fraction, decay):
    release = release_from_voltage(voltage, baseline, fraction)
    baseline.mul_(decay).add_(voltage, alpha=1-decay)
    return release


def select_fraction(rows):
    for row in rows:
        if (row['finite'] and row['blank_rate'] < .01
                and row['peak_fraction'] <= .20
                and row['on_excess'] >= 5 and row['off_excess'] >= 5):
            return row['fraction']
    return None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptive', action='store_true',
                        help='use a local 250 ms moving voltage baseline')
    args = parser.parse_args()
    output_path = ADAPTIVE_OUT if args.adaptive else OUT
    raw_path = ADAPTIVE_RAW if args.adaptive else RAW
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    retina = Retina(**metadata['retina'])
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    types = np.asarray(metadata['retina']['cell_types'])
    ids = np.asarray(metadata['graph']['body_ids'], dtype=np.int64)
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'type', 'somaSide', 'status'])
    targets = np.array(sorted(int(row['bodyId']) for row in table.to_pylist()
                              if row['type'] == 'LC11' and row['somaSide'] == 'R'
                              and row['status'] == 'Traced'), dtype=np.int64)
    pre, post, contacts = collect_target_edges(WEIGHTS, targets)
    object_ids = ids[np.isin(types, SOURCE_TYPES)]
    retain = np.isin(pre, object_ids) & np.isin(post, targets)
    pre, post, contacts = pre[retain], post[retain], contacts[retain]
    sources = np.unique(pre)
    source_nodes = np.searchsorted(ids, sources)
    if len(targets) != 75 or len(sources) < 2000 or not np.all(
            graph.signs[source_nodes] == 1):
        raise AssertionError('unexpected object-pathway anatomy or signs')
    config = NeuronConfig(**metadata['neurons'])
    shadow = ShadowLC11(np.searchsorted(sources, pre),
                        np.searchsorted(targets, post),
                        contacts.astype(np.float32)*metadata['gain'],
                        len(sources), len(targets), config)
    net = MultiContextEfficacyNetwork(
        graph, metadata['delays'], metadata['pathways'], config=config,
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    zero = torch.zeros_like(net.voltage)
    node_indices = torch.tensor(source_nodes)
    baseline_decay = math.exp(-config.dt/BASELINE_TAU)
    warm_baseline = None
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
        voltage = net.voltage[0, node_indices]
        if warm_baseline is None:
            warm_baseline = voltage.clone()
        else:
            warm_baseline.mul_(baseline_decay).add_(
                voltage, alpha=1-baseline_decay)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    warm_tick = net.step_index
    rest_voltage = net.voltage[0, source_nodes].clone()

    def capture(images, polarity):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        voltage = torch.empty((144, len(sources)))
        spikes = torch.empty_like(voltage)
        pixel_events = 0
        step = 0
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                voltage[step] = net.voltage[0, node_indices]
                spikes[step] = activity.spikes[0, node_indices].float()
                step += 1
        return dict(voltage=voltage, spikes=spikes,
                    pixel_events=pixel_events)

    def readout(captured, fraction):
        shadow.reset()
        local_baseline = warm_baseline.clone()
        spikes = torch.empty((144, len(targets)), dtype=torch.bool)
        currents = torch.empty((144, len(targets)))
        releases = torch.empty_like(captured['voltage'])
        for tick in range(144):
            if fraction is None:
                release = captured['spikes'][tick]
            elif args.adaptive:
                release = adaptive_release(captured['voltage'][tick],
                                           local_baseline, fraction,
                                           baseline_decay)
            else:
                release = release_from_voltage(captured['voltage'][tick],
                                               rest_voltage, fraction)
            spikes[tick] = shadow.step(release)
            currents[tick] = shadow.current
            releases[tick] = release
        finite = bool(torch.isfinite(currents).all()
                      and torch.isfinite(shadow.voltage).all())
        return dict(spikes=spikes, currents=currents, releases=releases,
                    finite=finite, pixel_events=captured['pixel_events'])

    def compare(observed, blank):
        excess = observed['spikes'][TRANSIT].int()-blank['spikes'][TRANSIT].int()
        current_delta = (observed['currents'][TRANSIT] -
                         blank['currents'][TRANSIT]).abs()
        return dict(excess_spikes=int(excess.sum()),
                    spikes=int(observed['spikes'][TRANSIT].sum()),
                    blank_spikes=int(blank['spikes'][TRANSIT].sum()),
                    cells_with_excess=int((excess.sum(0) > 0).sum()),
                    mean_absolute_current_change=float(current_delta.mean()),
                    peak_fraction=float(observed['spikes'][TRANSIT].float().mean(1).max()),
                    finite=observed['finite'] and blank['finite'],
                    pixel_events=observed['pixel_events'])

    blank_images = {polarity: [torch.full((1, 32, 64),
                    polarity == 'off', dtype=torch.bool)]*18
                    for polarity in ('on', 'off')}
    calibration_inputs = {polarity: dict(
        blank=capture(blank_images[polarity], polarity),
        dot=capture(frames_for_condition(18, 1, polarity), polarity))
        for polarity in ('on', 'off')}
    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  weights_sha256=checksum(WEIGHTS),
                  graph_sha256=graph.identity(),
                  lc11_neurons=len(targets), source_neurons=len(sources),
                  measured_edges=len(pre), measured_contacts=int(contacts.sum()),
                  voltage_scale=VOLTAGE_SCALE, fractions=FRACTIONS,
                  baseline_mode='adaptive-250ms' if args.adaptive else 'fixed-warm',
                  calibration=[], selected_fraction=None, spike_only={}, frozen={})
    report['blank_voltage_drift'] = {}
    for polarity, pair in calibration_inputs.items():
        drift = pair['blank']['voltage']-rest_voltage
        positive = drift.clamp(min=0)
        report['blank_voltage_drift'][polarity] = dict(
            mean_positive=float(positive.mean()),
            p99_positive=float(torch.quantile(positive.flatten(), .99)),
            source_cells_above_scale=int((positive.max(0).values >=
                                          VOLTAGE_SCALE).sum()),
            source_spikes=int(pair['blank']['spikes'].sum()))
    traces = {}
    for polarity in ('on', 'off'):
        pair = calibration_inputs[polarity]
        spike_dot = readout(pair['dot'], None)
        spike_blank = readout(pair['blank'], None)
        report['spike_only'][polarity] = compare(spike_dot, spike_blank)
    for fraction in FRACTIONS:
        values = {}
        blank_release_means = []
        for polarity in ('on', 'off'):
            pair = calibration_inputs[polarity]
            dot = readout(pair['dot'], fraction)
            blank = readout(pair['blank'], fraction)
            values[polarity] = compare(dot, blank)
            blank_release_means.append(float(blank['releases'].mean()))
            for condition, result in (('dot', dot), ('blank', blank)):
                key = f'calibration/{fraction}/{polarity}/{condition}'
                traces[f'{key}/release'] = result['releases'].numpy()
                traces[f'{key}/current'] = result['currents'].numpy()
                traces[f'{key}/spikes'] = result['spikes'].numpy()
        row = dict(fraction=fraction,
                   on_excess=values['on']['excess_spikes'],
                   off_excess=values['off']['excess_spikes'],
                   blank_rate=max(values[p]['blank_spikes']
                                  for p in ('on', 'off'))/(len(targets)*96),
                   peak_fraction=max(values[p]['peak_fraction']
                                     for p in ('on', 'off')),
                   blank_release_mean=max(blank_release_means),
                   finite=all(values[p]['finite'] for p in ('on', 'off')),
                   on=values['on'], off=values['off'])
        report['calibration'].append(row)
        print(json.dumps(dict(calibration=row)), flush=True)
    selected = select_fraction(report['calibration'])
    report['selected_fraction'] = selected
    if selected is not None:
        for center in (18, 46):
            for polarity in ('on', 'off'):
                baseline = (calibration_inputs[polarity]['blank'] if center == 18
                            else capture(blank_images[polarity], polarity))
                blank = readout(baseline, selected)
                for kind in ('dot', 'bar', 'static'):
                    for direction in ((1, -1) if kind != 'static' else (1,)):
                        images = (static_dot(center, polarity) if kind == 'static'
                                  else frames_for_condition(center, direction,
                                                            polarity, kind=kind))
                        captured = (calibration_inputs[polarity]['dot']
                                    if center == 18 and kind == 'dot'
                                    and direction == 1 else capture(images, polarity))
                        result = readout(captured, selected)
                        key = f'{center}-{polarity}-{kind}-{direction:+d}'
                        report['frozen'][key] = compare(result, blank)
                        traces[f'{key}/release'] = result['releases'].numpy()
                        traces[f'{key}/current'] = result['currents'].numpy()
                        traces[f'{key}/spikes'] = result['spikes'].numpy()
                        print(json.dumps(dict(condition=key,
                              result=report['frozen'][key])), flush=True)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **traces)
    report['per_tick_file'] = str(raw_path)
    report['per_tick_sha256'] = checksum(raw_path)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output_path), selected_fraction=selected,
                          frozen_cases=len(report['frozen']))), flush=True)


if __name__ == '__main__':
    main()
