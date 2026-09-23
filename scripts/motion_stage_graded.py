"""Frozen flash calibration and bounded motion check for opt-in graded Mi4."""
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from full_context_pong import NETWORK_STATE
from graded_mi4 import GradedMi4Network, blank_limited_release
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_timing import frames as timing_frames
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina


OUT = Path('runs/motion-stage-signal-integration-v1')
VOLTAGE_SCALE = .0042
FRACTIONS = (.25, .5, 1.)
BLANK_CAP = .05
FLASH_MINIMUM = .005
GRADED_STATE = ('graded_history', 'graded_baseline', 'graded_current',
                'last_graded_impulse')


def select_fraction(rows):
    for row in rows:
        if (row['finite'] and row['blank_p99_current'] <= BLANK_CAP
                and min(row['flash_modulation'].values()) >= FLASH_MINIMUM):
            return row['release_fraction']
    return None


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    m = payload['metadata']
    retina = Retina(**m['retina'])
    graph = Graph(**m['graph'], gain=m['gain'])
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    inferred, _ = infer_columns(m, retina, table)
    types = np.asarray(m['retina']['cell_types'])
    t4 = torch.tensor(np.char.startswith(types.astype(str), 'T4'))
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')

    def masks(center, halfwidth):
        path = ((x >= center-halfwidth) & (x <= center+halfwidth)
                & (y >= 9) & (y <= 23))
        bins = retina.pixel_bins[path.flatten()].unique().numpy()
        return {label: torch.tensor((types == label) & np.isin(inferred, bins))
                for label in ('T4a', 'T4b', 'T4c', 'T4d')}

    def make_network(shuffle=False):
        net = GradedMi4Network(graph, m['delays'], m['pathways'],
            config=NeuronConfig(**m['neurons']), cell_types=m['retina']['cell_types'],
            target_mask=retina.injected, base_release=.001,
            voltage_scale=VOLTAGE_SCALE, release_fraction=FRACTIONS[0], shuffle=shuffle)
        net.set_weights(payload['state']['network']['magnitudes'].clamp(0,
                        m['learning']['maximum_weight']))
        return net

    net = make_network()
    release = blank_limited_release(net, cap=.045)
    zero = torch.zeros_like(net.voltage)
    for _ in range(m['config']['warmup_steps']):
        net.step(zero)
    ungraded = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    ungraded_tick = net.step_index

    def warm_grading(network, fraction, base_release, initial, tick):
        for name, value in initial.items():
            getattr(network, name).copy_(value)
        network.step_index = tick
        network.graded_enabled = False
        network.configure_release(base_release, VOLTAGE_SCALE, fraction)
        network.enable_graded()
        blank_levels = []
        for _ in range(m['config']['warmup_steps']):
            network.step(zero)
            blank_levels.append(network.graded_current[0, t4].abs().clone())
        snapshot = {name: getattr(network, name).clone()
                    for name in NETWORK_STATE+GRADED_STATE}
        return snapshot, network.step_index, float(torch.quantile(
            torch.stack(blank_levels)[-128:].flatten(), .99))

    def probe(network, warm, tick0, images, polarity, *, trace=False, center=None):
        for name, state in warm.items():
            getattr(network, name).copy_(state)
        network.step_index = tick0
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        currents = []
        spikes = torch.zeros(network.n, dtype=torch.int32)
        peak_fraction = 0.
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*m['config']['sensory_gain']
            for neural_tick in range(8):
                activity = network.step(injection if neural_tick == 0 else zero)
                if trace:
                    currents.append(network.graded_current[0, t4].clone())
                if center is not None and 3 <= frame < 15:
                    spikes.add_(activity.spikes[0].int())
                    peak_fraction = max(peak_fraction,
                                        float(activity.spikes.float().mean()))
        return dict(currents=torch.stack(currents) if trace else None,
                    spikes=spikes, peak_fraction=peak_fraction,
                    finite=bool(torch.isfinite(network.voltage).all()
                                and torch.isfinite(network.graded_current).all()))

    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  graph_sha256=graph.identity(), voltage_scale=VOLTAGE_SCALE,
                  blank_cap=BLANK_CAP, flash_minimum=FLASH_MINIMUM,
                  analytic_base_release=release, graded_edges=int(len(net.graded_edges)),
                  calibration=[], selected_fraction=None, motion={})
    for fraction in FRACTIONS:
        calibrated_release = release
        for _ in range(4):
            warm, tick, blank_p99 = warm_grading(net, fraction, calibrated_release,
                                                ungraded, ungraded_tick)
            if blank_p99 <= BLANK_CAP:
                break
            calibrated_release *= .045/blank_p99
        modulation = {}
        finite = True
        for center in (18, 46):
            blank = probe(net, warm, tick, timing_frames(center, 'blank'), 'on', trace=True)
            flash = probe(net, warm, tick, timing_frames(center, 'flash'), 'on', trace=True)
            local = torch.stack([mask[t4] for mask in masks(center, 5).values()]).any(0)
            difference = (flash['currents'][:, local]-blank['currents'][:, local]).abs()
            modulation[str(center)] = float(difference[16:].mean(1).max())
            finite = finite and flash['finite'] and blank['finite']
        row = dict(release_fraction=fraction, base_release=calibrated_release,
                   blank_p99_current=blank_p99,
                   flash_modulation=modulation, finite=finite)
        report['calibration'].append(row)
        print(json.dumps(dict(calibration=row)), flush=True)
    selected = select_fraction(report['calibration'])
    report['selected_fraction'] = selected
    if selected is not None:
        selected_release = next(row['base_release'] for row in report['calibration']
                                if row['release_fraction'] == selected)
        warm, tick, blank_p99 = warm_grading(net, selected, selected_release,
                                            ungraded, ungraded_tick)
        for center in (18, 46):
            local = masks(center, 8)
            local_t4 = torch.stack([mask[t4] for mask in local.values()]).any(0)
            for polarity in ('on', 'off'):
                blank_image = torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)
                baseline = probe(net, warm, tick, [blank_image]*18, polarity,
                                 trace=True, center=center)
                outcomes = {}
                for direction in (1, -1):
                    result = probe(net, warm, tick,
                        frames_for_condition(center, direction, polarity, kind='bar'),
                        polarity, trace=True, center=center)
                    current_difference = (result['currents'][24:120, local_t4]
                                          -baseline['currents'][24:120, local_t4]).abs()
                    outcomes['right' if direction == 1 else 'left'] = dict(
                        spikes={label: int(result['spikes'][mask].sum())
                                for label, mask in local.items()},
                        excess={label: int((result['spikes'][mask]
                                            -baseline['spikes'][mask]).sum())
                                for label, mask in local.items()},
                        mean_absolute_graded_current_change=float(current_difference.mean()),
                        peak_graded_current_change=float(current_difference.mean(1).max()),
                        finite=result['finite'], peak_fraction=result['peak_fraction'])
                report['motion'][f'{center}-{polarity}'] = dict(
                    local_neurons={label: int(mask.sum()) for label, mask in local.items()},
                    blank_spikes={label: int(baseline['spikes'][mask].sum())
                                  for label, mask in local.items()}, **outcomes)
        shuffled = make_network(shuffle=True)
        shuffled_warm, shuffled_tick, _ = warm_grading(
            shuffled, selected, selected_release, ungraded, ungraded_tick)
        report['shuffled_on'] = {}
        for center in (18, 46):
            local = masks(center, 8)
            blank_image = torch.zeros(1, 32, 64, dtype=torch.bool)
            baseline = probe(shuffled, shuffled_warm, shuffled_tick,
                             [blank_image]*18, 'on', center=center)
            cases = {}
            for direction in (1, -1):
                result = probe(shuffled, shuffled_warm, shuffled_tick,
                    frames_for_condition(center, direction, 'on', kind='bar'),
                    'on', center=center)
                cases['right' if direction == 1 else 'left'] = {
                    label: int((result['spikes'][mask]-baseline['spikes'][mask]).sum())
                    for label, mask in local.items()}
            report['shuffled_on'][str(center)] = cases
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT/'graded-calibration-and-motion.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), selected_fraction=selected,
                          motion=report['motion'])), flush=True)


if __name__ == '__main__':
    main()
