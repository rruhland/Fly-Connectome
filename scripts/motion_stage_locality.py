"""Offline receptive-column audit of saved frozen motion responses."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from fly_connectome.data import checksum
from fly_connectome.sensor import Retina


SOURCE = Path('checkpoints/event-v1-combined-rate-initial.pt')
ANNOTATIONS = Path('data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather')
RESPONSES = Path('runs/motion-stage-audit-v1/per-neuron-responses.pt')
OUT = Path('runs/motion-stage-audit-v1/locality.json')


def infer_columns(metadata, retina, annotations):
    graph = metadata['graph']
    ids = np.asarray(graph['body_ids'])
    pre, post, contacts = (np.asarray(graph[k]) for k in ('pre', 'post', 'contacts'))
    types = np.asarray(metadata['retina']['cell_types'])
    body, q, r = (annotations[k].to_numpy(zero_copy_only=False)
                  for k in ('bodyId', 'assignedOlHex1', 'assignedOlHex2'))
    locations = {int(b): (float(x), float(y)) for b, x, y in zip(body, q, r)
                 if np.isfinite(x) and np.isfinite(y)}
    pre_q = np.array([locations.get(int(b), (np.nan, np.nan))[0] for b in ids])
    pre_r = np.array([locations.get(int(b), (np.nan, np.nan))[1] for b in ids])
    targets = np.char.startswith(types.astype(str), 'T4') | np.char.startswith(types.astype(str), 'T5')
    afferents = ((np.char.startswith(types[post].astype(str), 'T4') & (types[pre] == 'Mi1'))
                 | (np.char.startswith(types[post].astype(str), 'T5')
                    & np.isin(types[pre], ['Tm1', 'Tm2'])))
    valid = afferents & np.isfinite(pre_q[pre]) & np.isfinite(pre_r[pre])
    mass = np.bincount(post[valid], weights=contacts[valid], minlength=len(ids))
    mean_q = np.bincount(post[valid], weights=contacts[valid]*pre_q[pre[valid]],
                         minlength=len(ids))/np.maximum(mass, 1)
    mean_r = np.bincount(post[valid], weights=contacts[valid]*pre_r[pre[valid]],
                         minlength=len(ids))/np.maximum(mass, 1)
    inferred = np.full(len(ids), -1, dtype=np.int64)
    hexes = np.asarray(retina.spec['hex_columns'])
    centers = np.column_stack((hexes[:, 0]+hexes[:, 1]/2,
                               hexes[:, 1]*math.sqrt(3)/2))
    good = np.flatnonzero(targets & (mass > 0))
    for batch in np.array_split(good, max(1, math.ceil(len(good)/500))):
        points = np.column_stack((mean_q[batch]+mean_r[batch]/2,
                                  mean_r[batch]*math.sqrt(3)/2))
        inferred[batch] = ((points[:, None, :]-centers[None, :, :])**2).sum(2).argmin(1)
    return inferred, mass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stimulus', choices=('dot', 'bar'), default='dot')
    parser.add_argument('--recovery', action='store_true',
                        help='read the approved opt-in recovery response instead of baseline')
    args = parser.parse_args()
    directory = (Path('runs/motion-stage-recovery-v1') if args.recovery else RESPONSES.parent)
    responses_path = directory/('per-neuron-responses.pt' if args.stimulus == 'dot'
                                else 'bar-per-neuron-responses.pt')
    output_path = directory/('locality.json' if args.stimulus == 'dot'
                             else 'bar-locality.json')
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    retina = Retina(**metadata['retina'])
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    inferred, mass = infer_columns(metadata, retina, table)
    saved = torch.load(responses_path, weights_only=True)
    types = np.asarray(saved['cell_types'])
    output = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  method='contact-weighted centroid of direct Mi1 to T4 or Tm1/Tm2 to T5 inputs',
                  groups={}, direction_contrasts={})
    for label in ('T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'):
        kind = types == label
        output['groups'][label] = dict(neurons=int(kind.sum()),
            inferred_columns=int(((inferred >= 0) & kind).sum()),
            median_afferent_contacts=float(np.median(mass[kind])))
        for center in (18, 46):
            path = (torch.arange(64)[None, :] >= center-8) & (
                torch.arange(64)[None, :] <= center+8)
            path = path.expand(32, -1) & (torch.arange(32)[:, None] >= 13) & (
                torch.arange(32)[:, None] <= 19)
            bins = retina.pixel_bins[path.flatten()].unique().numpy()
            local = kind & np.isin(inferred, bins)
            row = output['groups'][label].setdefault(str(center),
                dict(local_neurons=int(local.sum())))
            for polarity in ('on', 'off'):
                baseline = saved['baselines'][polarity]['spikes'].numpy()
                values = []
                for direction in (1, -1):
                    key = f'{center}-{polarity}-{direction:+d}'
                    spikes = saved['responses'][key]['spikes'].numpy()
                    values.append(dict(spikes=int(spikes[local].sum()),
                                       blank_spikes=int(baseline[local].sum()),
                                       changed_cells=int((spikes[local] != baseline[local]).sum()),
                                       excess=int((spikes[local]-baseline[local]).sum())))
                right_v = saved['responses'][f'{center}-{polarity}-+1']['mean_voltage'].numpy()
                left_v = saved['responses'][f'{center}-{polarity}--1']['mean_voltage'].numpy()
                difference = right_v[local]-left_v[local]
                row[polarity] = dict(right=values[0], left=values[1],
                                     right_minus_left_excess=values[0]['excess']-values[1]['excess'],
                                     mean_voltage_right_minus_left=float(difference.mean()),
                                     mean_absolute_voltage_difference=float(np.abs(difference).mean()),
                                     voltage_positive_cells=int((difference > 0).sum()),
                                     voltage_negative_cells=int((difference < 0).sum()))
        for polarity in ('on', 'off'):
            values = [output['groups'][label][str(center)][polarity]['right_minus_left_excess']
                      for center in (18, 46)]
            output['direction_contrasts'][f'{label}:{polarity}'] = dict(
                right_minus_left=values, sign_consistent=values[0]*values[1] > 0)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output_path), groups=output['groups'],
                          direction_contrasts=output['direction_contrasts'])), flush=True)


if __name__ == '__main__':
    main()
