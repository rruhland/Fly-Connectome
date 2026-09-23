"""Read-only partner-column coverage for the missing measured CT1 route."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import torch

from fly_connectome.data import checksum
from fly_connectome.sensor import Retina
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns
from motion_ct1_wiring import WEIGHTS


OUT = Path('docs/experiments/2026-09-23-ct1-column-coverage-results.json')
CT1 = 10009
SOURCES = ('Tm1', 'Tm9')
TARGETS = ('T5c', 'T5d')


def coverage(output, inputs, hexes, bins):
    source = inputs.sum(0) > 0
    difference = hexes[:, None, :]-hexes[None, :, :]
    distance = np.maximum.reduce((abs(difference[:, :, 0]),
                                  abs(difference[:, :, 1]),
                                  abs(difference.sum(2))))
    nearby = (distance <= 1) @ source.astype(np.int32) > 0
    mass = output[bins].sum()
    return dict(contacts=int(mass), columns=int((output[bins] > 0).sum()),
        same_column_fraction=float(output[bins][source[bins]].sum()/mass) if mass else 0.,
        same_or_neighbor_fraction=float(output[bins][nearby[bins]].sum()/mass) if mass else 0.)


def main():
    metadata = torch.load(SOURCE, weights_only=True)['metadata']
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    ids = np.asarray(metadata['graph']['body_ids'])
    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    source_columns = annotated_columns(metadata, retina, annotations, classes=SOURCES)
    target_columns, _ = infer_columns(metadata, retina, annotations)
    ncols = len(retina.spec['hex_columns'])
    source_mass = np.zeros((2, 2, ncols), dtype=np.int64)
    target_mass = np.zeros((2, 2, ncols), dtype=np.int64)
    totals = dict(input_edges=0, input_contacts=0, input_edges_ge3=0,
                  input_contacts_ge3=0, output_edges=0, output_contacts=0,
                  output_edges_ge3=0, output_contacts_ge3=0,
                  unmapped_input_contacts=0, unmapped_output_contacts=0)
    source_info = {int(ids[i]): (SOURCES.index(types[i]), int(source_columns[i]))
                   for i in np.flatnonzero(np.isin(types, SOURCES))}
    target_info = {int(ids[i]): (TARGETS.index(types[i]), int(target_columns[i]))
                   for i in np.flatnonzero(np.isin(types, TARGETS))}
    # The original file stores one weighted row per body pair, not per synapse.
    with pa.memory_map(str(WEIGHTS), 'r') as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = batch.column('body_pre').to_numpy()
            post = batch.column('body_post').to_numpy()
            weight = batch.column('weight').to_numpy()
            incoming = np.flatnonzero(post == CT1)
            outgoing = np.flatnonzero(pre == CT1)
            for edge in incoming:
                info = source_info.get(int(pre[edge]))
                if info is None:
                    continue
                group, column = info
                count = int(weight[edge])
                totals['input_edges'] += 1
                totals['input_contacts'] += count
                if column < 0:
                    totals['unmapped_input_contacts'] += count
                else:
                    source_mass[0, group, column] += count
                if count >= 3:
                    totals['input_edges_ge3'] += 1
                    totals['input_contacts_ge3'] += count
                    if column >= 0:
                        source_mass[1, group, column] += count
            for edge in outgoing:
                info = target_info.get(int(post[edge]))
                if info is None:
                    continue
                group, column = info
                count = int(weight[edge])
                totals['output_edges'] += 1
                totals['output_contacts'] += count
                if column < 0:
                    totals['unmapped_output_contacts'] += count
                else:
                    target_mass[0, group, column] += count
                if count >= 3:
                    totals['output_edges_ge3'] += 1
                    totals['output_contacts_ge3'] += count
                    if column >= 0:
                        target_mass[1, group, column] += count
    hexes = np.asarray(retina.spec['hex_columns'])
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    fields = {'all': np.arange(ncols)}
    for center in (10, 22, 16):
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        fields[f'y{center}'] = retina.pixel_bins[field.flatten()].unique().numpy()
    report = dict(source_sha256=checksum(SOURCE),
        annotations_sha256=checksum(ANNOTATIONS),
        weights_sha256=checksum(WEIGHTS), ct1_body=CT1,
        source_classes=SOURCES, target_classes=TARGETS,
        inferred_t5_columns=True, totals=totals, screens={})
    for threshold, label in enumerate(('all_edges', 'edges_ge3')):
        rows = report['screens'][label] = {}
        for target_index, target in enumerate(TARGETS):
            target_rows = rows[target] = {}
            for field, bins in fields.items():
                target_rows[field] = {
                    name: coverage(target_mass[threshold, target_index],
                                   source_mass[threshold, indices], hexes, bins)
                    for name, indices in (('Tm1', slice(0, 1)),
                                          ('Tm9', slice(1, 2)),
                                          ('either', slice(None)))}
        rows['source_contacts'] = {name: int(source_mass[threshold, i].sum())
                                   for i, name in enumerate(SOURCES)}
        rows['target_contacts'] = {name: int(target_mass[threshold, i].sum())
                                   for i, name in enumerate(TARGETS)}
    selected = report['screens']['edges_ge3']
    report['passes_body_level_coverage_screen'] = bool(
        totals['unmapped_input_contacts'] < .2*totals['input_contacts']
        and totals['unmapped_output_contacts'] < .2*totals['output_contacts']
        and all(selected[target][field]['either']['contacts'] > 0
                and selected[target][field]['either']['same_or_neighbor_fraction'] >= .8
                for target in TARGETS for field in ('y10', 'y22', 'y16'))
        and all(selected[target]['y16'][source]['same_or_neighbor_fraction'] > 0
                for target in TARGETS for source in SOURCES))
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), totals=totals,
        passes_body_level_coverage_screen=report['passes_body_level_coverage_screen'],
        holdout={target: selected[target]['y16']['either'] for target in TARGETS})),
        flush=True)


if __name__ == '__main__':
    main()
