"""Opt-in frozen T3 rest-current calibration for small moving targets."""
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina


OUT = Path('runs/motion-t3-target-v1')
CURRENTS = (0., .85, .95, 1.05, 1.15)


def infer_t3_columns(metadata, retina, annotations):
    graph = metadata['graph']
    types = np.asarray(metadata['retina']['cell_types'])
    ids = np.asarray(graph['body_ids'])
    pre, post, contacts = (np.asarray(graph[k]) for k in ('pre', 'post', 'contacts'))
    body, q, r = (annotations[k].to_numpy(zero_copy_only=False)
                  for k in ('bodyId', 'assignedOlHex1', 'assignedOlHex2'))
    positions = {int(b): (float(x), float(y)) for b, x, y in zip(body, q, r)
                 if np.isfinite(x) and np.isfinite(y)}
    source = np.asarray([positions.get(int(b), (np.nan, np.nan)) for b in ids])
    selected = ((types[post] == 'T3') & np.isin(types[pre], ('Mi1', 'Tm1'))
                & np.isfinite(source[pre]).all(1))
    mass = np.bincount(post[selected], weights=contacts[selected], minlength=len(ids))
    centers = np.column_stack([
        np.bincount(post[selected], weights=contacts[selected]*source[pre[selected], axis],
                    minlength=len(ids))/np.maximum(mass, 1)
        for axis in (0, 1)])
    hexes = np.asarray(retina.spec['hex_columns'])
    axes = np.column_stack((hexes[:, 0]+hexes[:, 1]/2,
                            hexes[:, 1]*math.sqrt(3)/2))
    inferred = np.full(len(ids), -1, dtype=np.int64)
    good = np.flatnonzero((types == 'T3') & (mass > 0))
    for batch in np.array_split(good, max(1, math.ceil(len(good)/500))):
        xy = np.column_stack((centers[batch, 0]+centers[batch, 1]/2,
                              centers[batch, 1]*math.sqrt(3)/2))
        inferred[batch] = ((xy[:, None, :]-axes[None, :, :])**2).sum(2).argmin(1)
    return inferred, mass


def select_current(rows):
    for row in rows:
        if (row['finite'] and row['blank_rate'] < .01
                and row['on_excess'] >= 5 and row['off_excess'] >= 5):
            return row['rest_current']
    return None


def static_dot(center, polarity):
    frames = frames_for_condition(center, 1, polarity)
    for frame in range(2, 15):
        frames[frame] = frames[8]
    return frames


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    m = payload['metadata']
    retina = Retina(**m['retina'])
    graph = Graph(**m['graph'], gain=m['gain'])
    net = MultiContextEfficacyNetwork(graph, m['delays'], m['pathways'],
        config=NeuronConfig(**m['neurons']), cell_types=m['retina']['cell_types'],
        target_mask=retina.injected)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(0,
                    m['learning']['maximum_weight']))
    initial = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    base_rest = net.rest_current.clone()
    zero = torch.zeros_like(net.voltage)
    types = np.asarray(m['retina']['cell_types'])
    t3 = torch.tensor(types == 'T3')
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, mass = infer_t3_columns(m, retina, table)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')

    def local_mask(center):
        path = ((x >= center-8) & (x <= center+8)
                & (y >= 13) & (y <= 19))
        bins = retina.pixel_bins[path.flatten()].unique().numpy()
        return torch.tensor((types == 'T3') & np.isin(columns, bins))

    def warm(current):
        net.rest_current.copy_(base_rest)
        net.rest_current[t3] = current
        for name, value in initial.items():
            getattr(net, name).copy_(value)
        net.step_index = 0
        for _ in range(m['config']['warmup_steps']):
            net.step(zero)
        return {name: getattr(net, name).clone() for name in NETWORK_STATE}, net.step_index

    def probe(warm_state, warm_tick, images, polarity):
        for name, value in warm_state.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        spikes = torch.zeros(net.n, dtype=torch.int32)
        event_count = 0
        peak_fraction = 0.
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*m['config']['sensory_gain']
            if 3 <= frame < 15:
                event_count += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                if 3 <= frame < 15:
                    spikes.add_(activity.spikes[0].int())
                    peak_fraction = max(peak_fraction, float(activity.spikes.float().mean()))
        return dict(spikes=spikes, pixel_events=event_count,
                    peak_fraction=peak_fraction,
                    finite=bool(torch.isfinite(net.voltage).all()))

    report = dict(source_sha256=source_sha,
        annotations_sha256=checksum(ANNOTATIONS), graph_sha256=graph.identity(),
        t3_neurons=int(t3.sum()), inferred_columns=int(((columns >= 0) & t3.numpy()).sum()),
        median_input_contacts=float(np.median(mass[t3.numpy()])),
        calibration=[], selected_current=None, frozen={})
    for current in CURRENTS:
        state, tick0 = warm(current)
        local = local_mask(18)
        results = {}
        for polarity in ('on', 'off'):
            blank_image = torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)
            blank = probe(state, tick0, [blank_image]*18, polarity)
            dot = probe(state, tick0, frames_for_condition(18, 1, polarity), polarity)
            results[polarity] = dict(excess=int((dot['spikes'][local]
                                                -blank['spikes'][local]).sum()),
                changed_cells=int((dot['spikes'][local]
                                   !=blank['spikes'][local]).sum()),
                local_spikes=int(dot['spikes'][local].sum()),
                blank_local_spikes=int(blank['spikes'][local].sum()),
                blank_global_spikes=int(blank['spikes'][t3].sum()),
                pixel_events=dot['pixel_events'],
                peak_fraction=max(dot['peak_fraction'], blank['peak_fraction']),
                finite=dot['finite'] and blank['finite'])
        row = dict(rest_current=current, local_neurons=int(local.sum()),
            on_excess=results['on']['excess'], off_excess=results['off']['excess'],
            blank_rate=max(results[p]['blank_global_spikes'] for p in ('on', 'off'))/(
                int(t3.sum())*12*8),
            finite=all(results[p]['finite'] for p in ('on', 'off')),
            on=results['on'], off=results['off'])
        report['calibration'].append(row)
        print(json.dumps(dict(calibration=row)), flush=True)
    selected = select_current(report['calibration'])
    report['selected_current'] = selected
    if selected is not None:
        state, tick0 = warm(selected)
        for center in (18, 46):
            local = local_mask(center)
            for polarity in ('on', 'off'):
                blank_image = torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)
                blank = probe(state, tick0, [blank_image]*18, polarity)
                for kind in ('dot', 'bar', 'static'):
                    for direction in ((1, -1) if kind != 'static' else (1,)):
                        images = (static_dot(center, polarity) if kind == 'static'
                                  else frames_for_condition(center, direction, polarity,
                                                            kind=kind))
                        result = probe(state, tick0, images, polarity)
                        report['frozen'][f'{center}-{polarity}-{kind}-{direction:+d}'] = dict(
                            local_neurons=int(local.sum()),
                            local_spikes=int(result['spikes'][local].sum()),
                            local_excess=int((result['spikes'][local]
                                              -blank['spikes'][local]).sum()),
                            blank_local_spikes=int(blank['spikes'][local].sum()),
                            global_spikes=int(result['spikes'][t3].sum()),
                            pixel_events=result['pixel_events'],
                            finite=result['finite'], peak_fraction=result['peak_fraction'])
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'t3-calibration-and-frozen.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path), selected_current=selected,
                          frozen=report['frozen'])), flush=True)


if __name__ == '__main__':
    main()
