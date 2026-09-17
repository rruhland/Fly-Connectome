"""Verify pinned release inputs, extract the preregistered roster, and write artifacts.

Run from repository root after installing .[data,test]. Raw inputs stay in data/raw.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from fly_connectome.data import Source, verify_source
from fly_connectome.extraction import Selection, extract_tables


def membership(values, sorted_ids):
    locations = np.searchsorted(sorted_ids, values)
    return (locations < len(sorted_ids)) & (sorted_ids[np.minimum(locations, len(sorted_ids) - 1)] == values)


def prepare(raw, source_manifest, output):
    raw, output = Path(raw), Path(output)
    provenance = json.loads(Path(source_manifest).read_text())
    for source in provenance['sources']:
        verify_source(raw / source['filename'], Source(source['filename'], source['url'], source['sha256']))
    print('Release checksums verified', flush=True)
    annotations = feather.read_table(raw / 'body-annotations-male-cns-v1.0-minconf-0.5.feather').to_pylist()
    annotations = [r for r in annotations if r.get('status') == 'Traced' and
        (r.get('superclass') in ('ol_intrinsic', 'visual_projection', 'visual_centrifugal', 'cb_intrinsic') or r.get('type') == 'DNa02')]
    ids = np.array(sorted(r['bodyId'] for r in annotations), dtype=np.int64)
    nt = feather.read_table(raw / 'body-neurotransmitters-male-cns-v1.0.feather')
    nt = nt.filter(pa.array(membership(nt['body'].to_numpy(), ids))).to_pylist()
    roi_by_id = {}
    with pa.memory_map(str(raw / 'Neuprint_Neurons.feather')) as source:
        schema = pa.ipc.open_file(source).schema
        options = pa.ipc.IpcReadOptions(included_fields=[schema.get_field_index('bodyId:long'), schema.get_field_index('roiInfo:string')])
        reader = pa.ipc.open_file(source, options=options)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            selected = batch.filter(pa.array(membership(batch['bodyId:long'].to_numpy(), ids)))
            for row in selected.to_pylist():
                roi_by_id[row['bodyId:long']] = json.loads(row['roiInfo:string'] or '{}')
            if i % 200 == 0:
                print(f'ROI batches {i}/{reader.num_record_batches}; annotated bodies {len(roi_by_id)}', flush=True)
    rows = []
    with pa.memory_map(str(raw / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather')) as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            keep = membership(batch['body_pre'].to_numpy(), ids) & membership(batch['body_post'].to_numpy(), ids)
            batch = batch.filter(pa.array(keep))
            if batch.num_rows:
                rows.append(np.column_stack([batch[c].to_numpy() for c in ('body_pre', 'body_post', 'weight')]))
    edges = np.concatenate(rows)
    print(f'Annotated candidate contacts: {len(edges)} ordered pairs', flush=True)
    rules = Selection()
    extracted = extract_tables(annotations, nt, roi_by_id, edges, rules)
    graph = extracted.graph
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'graph-t1.npz', body_ids=graph.body_ids, pre=graph.pre,
                        post=graph.post, contacts=graph.contacts, signs=graph.signs, gain=graph.gain)
    metadata = dict(schema_version=1, dataset='male-cns:v1.0', sources=provenance['sources'],
                    extraction_rules=asdict(rules), threshold=extracted.threshold,
                    graph_sha256=graph.identity(), body_ids=graph.body_ids.tolist(),
                    pathways=extracted.pathways, motor_up=extracted.motor_up, motor_down=extracted.motor_down,
                    retina=extracted.retina, positions=extracted.positions, regions=extracted.regions,
                    sign_evidence=extracted.sign_evidence,
                    stages=extracted.stages,
                    sign_model={'acetylcholine': 1, 'gaba': -1, 'glutamate': -1, 'histamine': -1},
                    delay_model={'steps': 1, 'basis': 'uniform fixed one-tick non-Pong-specific default'},
                    anatomy_reports=extracted.reports)
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({t: {k: v for k,v in report.items() if k in ('neurons', 'edges', 'contacts', 'reachable_motor_ids')}
                      for t,report in extracted.reports.items()}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw')
    parser.add_argument('--sources', default='data/malecns-v1.0-sources.json')
    parser.add_argument('--output', default='data/cache/milestone-1')
    args = parser.parse_args()
    prepare(args.raw, args.sources, args.output)
