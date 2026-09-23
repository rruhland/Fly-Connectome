"""Read-only MaleCNS neuPrint CT1 synapse-site geometry audit."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.feather as feather
from scipy.spatial import cKDTree
import torch

from fly_connectome.data import checksum
from fly_connectome.sensor import Retina
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns


URL = 'https://neuprint.janelia.org/api/custom/custom'
DATASET = 'male-cns:v1.0'
CT1 = 10009
CACHE = Path('runs/motion-ct1-synapse-geometry-v1/sites.npz')
OUT = Path('docs/experiments/2026-09-23-ct1-synapse-geometry-results.json')
EXPECTED = {'input': 32728, 'output': 37186}


def query(kind, body_ids, *, count_only):
    body_list = ','.join(map(str, body_ids))
    if kind == 'input':
        pattern = (f'(b:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->'
                   f'(pre:Synapse)-[:SynapsesTo]->(post:Synapse)'
                   f'<-[:Contains]-(:SynapseSet)<-[:Contains]-'
                   f'(a:Neuron {{bodyId:{CT1}}})')
    else:
        pattern = (f'(a:Neuron {{bodyId:{CT1}}})-[:Contains]->'
                   f'(:SynapseSet)-[:Contains]->(pre:Synapse)'
                   f'-[:SynapsesTo]->(post:Synapse)<-[:Contains]-'
                   f'(:SynapseSet)<-[:Contains]-(b:Neuron)')
    # A synapse can belong to several SynapseSets. Deduplicate the physical
    # pre/post site pair, rather than counting every graph traversal path.
    returns = ('count(DISTINCT [b.bodyId,pre.location,post.location]) AS n'
               if count_only else
               'DISTINCT b.bodyId AS partner, pre.location.x AS x_pre, '
               'pre.location.y AS y_pre, pre.location.z AS z_pre, '
               'post.location.x AS x_post, post.location.y AS y_post, '
               'post.location.z AS z_post')
    cypher = (f'MATCH {pattern} WHERE b.bodyId IN [{body_list}] '
              f'AND pre.confidence >= 0.5 AND post.confidence >= 0.5 '
              f'RETURN {returns}')
    request = Request(URL,
        data=json.dumps(dict(dataset=DATASET, cypher=cypher)).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=180) as response:
        payload = json.load(response)
    expected_columns = (['n'] if count_only else
        ['partner', 'x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post'])
    if payload['columns'] != expected_columns:
        raise AssertionError('unexpected neuPrint response columns')
    return payload['data']


def hex_match(source_columns, target_columns, hexes):
    source = hexes[source_columns]
    target = hexes[target_columns]
    delta = source-target
    return np.maximum.reduce((abs(delta[:, 0]), abs(delta[:, 1]),
                              abs(delta.sum(1)))) <= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts-only', action='store_true')
    args = parser.parse_args()
    source_sha = checksum(SOURCE)
    metadata = torch.load(SOURCE, weights_only=True)['metadata']
    ids = np.asarray(metadata['graph']['body_ids'])
    types = np.asarray(metadata['retina']['cell_types'])
    bodies = {kind: ids[mask].tolist() for kind, mask in (
        ('input', np.isin(types, ('Tm1', 'Tm9'))),
        ('output', np.isin(types, ('T5c', 'T5d'))))}
    if not CACHE.exists() or args.counts_only:
        counts = {kind: int(query(kind, selected, count_only=True)[0][0])
                  for kind, selected in bodies.items()}
        print(json.dumps(dict(dataset=DATASET, ct1=CT1, counts=counts)), flush=True)
        if counts != EXPECTED:
            raise AssertionError('neuPrint synapse counts differ from raw contact table')
        if args.counts_only:
            return
        arrays = {}
        for kind, selected in bodies.items():
            rows = np.asarray(query(kind, selected, count_only=False), dtype=np.int64)
            if rows.shape != (EXPECTED[kind], 7):
                raise AssertionError('neuPrint rows differ from counted synapses')
            arrays[f'{kind}_body'] = rows[:, 0]
            arrays[f'{kind}_xyz'] = rows[:, 4:7] if kind == 'input' else rows[:, 1:4]
            print(json.dumps(dict(fetched=kind, rows=len(rows))), flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **arrays)
    with np.load(CACHE) as saved:
        input_body = saved['input_body']
        input_xyz = saved['input_xyz']
        output_body = saved['output_body']
        output_xyz = saved['output_xyz']
    if len(input_body) != EXPECTED['input'] or len(output_body) != EXPECTED['output']:
        raise AssertionError('cached CT1 sites have the wrong shape')
    retina = Retina(**metadata['retina'])
    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    source_column = annotated_columns(metadata, retina, annotations,
                                      classes=('Tm1', 'Tm9'))
    target_column, _ = infer_columns(metadata, retina, annotations)
    source_by_body = {int(ids[i]): int(source_column[i])
                      for i in np.flatnonzero(np.isin(types, ('Tm1', 'Tm9')))}
    target_by_body = {int(ids[i]): int(target_column[i])
                      for i in np.flatnonzero(np.isin(types, ('T5c', 'T5d')))}
    class_by_body = {int(ids[i]): types[i]
                     for i in np.flatnonzero(np.isin(types, ('Tm1', 'Tm9', 'T5c', 'T5d')))}
    input_column = np.asarray([source_by_body[int(body)] for body in input_body])
    output_column = np.asarray([target_by_body[int(body)] for body in output_body])
    if (input_column < 0).any() or (output_column < 0).any():
        raise AssertionError('synapse partner lacks an optic column')
    distance, nearest = cKDTree(input_xyz).query(output_xyz, k=1)
    nearest_body = input_body[nearest]
    hexes = np.asarray(retina.spec['hex_columns'])
    observed = hex_match(input_column[nearest], output_column, hexes)
    source_ids = np.asarray(sorted(source_by_body))
    source_labels = np.asarray([source_by_body[int(body)] for body in source_ids])
    nearest_positions = np.searchsorted(source_ids, nearest_body)
    if not np.array_equal(source_ids[nearest_positions], nearest_body):
        raise AssertionError('nearest source body lookup failed')
    rng = np.random.default_rng(20260923)
    shuffled = np.asarray([hex_match(source_labels[rng.permutation(
        len(source_labels))][nearest_positions], output_column, hexes)
        for _ in range(100)])
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    fields = {'all': np.arange(len(hexes))}
    for center in (10, 22, 16):
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        fields[f'y{center}'] = retina.pixel_bins[field.flatten()].unique().numpy()
    report = dict(dataset=DATASET, source_url=URL, source_sha256=source_sha,
        annotations_sha256=checksum(ANNOTATIONS),
        cache_sha256=checksum(CACHE), ct1_body=CT1, counts=EXPECTED,
        coordinate_units='8 nm voxels', nearest_site_k=1,
        shuffle_seed=20260923, shuffle_count=100, groups={})
    for subtype in ('T5c', 'T5d'):
        subtype_mask = np.asarray([class_by_body[int(body)] == subtype
                                   for body in output_body])
        rows = report['groups'][subtype] = {}
        for field, bins in fields.items():
            keep = subtype_mask & np.isin(output_column, bins)
            if not keep.any():
                raise AssertionError('empty CT1 output field')
            distances = distance[keep]
            null = shuffled[:, keep].mean(1)
            rows[field] = dict(contacts=int(keep.sum()),
                nearest_input_distance_voxels=dict(
                    p05=float(np.quantile(distances, .05)),
                    median=float(np.median(distances)),
                    p95=float(np.quantile(distances, .95))),
                within_5um_fraction=float((distances <= 625).mean()),
                nearest_same_or_neighbor_fraction=float(observed[keep].mean()),
                shuffled_fraction=dict(mean=float(null.mean()),
                    p95=float(np.quantile(null, .95)),
                    maximum=float(null.max())),
                nearest_source_class={name: int(sum(
                    class_by_body[int(body)] == name
                    for body in nearest_body[keep]))
                    for name in ('Tm1', 'Tm9')})
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        holdout={name: row['y16'] for name, row in report['groups'].items()})),
        flush=True)


if __name__ == '__main__':
    main()
