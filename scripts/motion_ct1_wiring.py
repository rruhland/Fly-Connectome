"""Read-only CT1 contacts into the pinned T5 roster from raw MaleCNS."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import torch

from fly_connectome.data import checksum
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS


WEIGHTS = Path('data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather')
TRANSMITTERS = Path('data/raw/body-neurotransmitters-male-cns-v1.0.feather')
OUT = Path('docs/experiments/2026-09-23-ct1-roster-audit-results.json')


def main():
    metadata = torch.load(SOURCE, weights_only=True)['metadata']
    graph = metadata['graph']
    ids = np.asarray(graph['body_ids'])
    types = np.asarray(metadata['retina']['cell_types'])
    t5 = ids[np.char.startswith(types.astype(str), 'T5')]
    tm = ids[np.isin(types, ('Tm1', 'Tm9'))]
    annotation = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'type', 'instance', 'somaSide', 'status'])
    ct1 = annotation.filter(pc.equal(annotation['type'], 'CT1')).to_pylist()
    ct1 = [row for row in ct1 if row['status'] == 'Traced']
    transmitter = feather.read_table(TRANSMITTERS)
    nt = {row['body']: row for row in transmitter.filter(pc.is_in(
        transmitter['body'], value_set=pa.array([row['bodyId'] for row in ct1]))).to_pylist()}
    counts = {row['bodyId']: dict(to_selected_t5_edges=0,
        to_selected_t5_contacts=0, to_selected_t5_edges_ge3=0,
        from_selected_tm1_tm9_edges=0, from_selected_tm1_tm9_contacts=0,
        from_selected_tm1_tm9_edges_ge3=0) for row in ct1}
    with pa.memory_map(str(WEIGHTS), 'r') as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = batch.column('body_pre').to_numpy()
            post = batch.column('body_post').to_numpy()
            weight = batch.column('weight').to_numpy()
            for body, row in counts.items():
                outgoing = (pre == body) & np.isin(post, t5)
                incoming = (post == body) & np.isin(pre, tm)
                row['to_selected_t5_edges'] += int(outgoing.sum())
                row['to_selected_t5_contacts'] += int(weight[outgoing].sum())
                row['to_selected_t5_edges_ge3'] += int((weight[outgoing] >= 3).sum())
                row['from_selected_tm1_tm9_edges'] += int(incoming.sum())
                row['from_selected_tm1_tm9_contacts'] += int(weight[incoming].sum())
                row['from_selected_tm1_tm9_edges_ge3'] += int((weight[incoming] >= 3).sum())
    report = dict(source_sha256=checksum(SOURCE),
        annotations_sha256=checksum(ANNOTATIONS), weights_sha256=checksum(WEIGHTS),
        transmitters_sha256=checksum(TRANSMITTERS),
        selected_t5=len(t5), selected_tm1_tm9=len(tm), ct1={})
    for row in ct1:
        body = row['bodyId']
        report['ct1'][str(body)] = dict(instance=row['instance'], soma_side=row['somaSide'],
            selected=bool(np.isin(body, ids)),
            transmitter_ground_truth=nt[body]['ground_truth'],
            transmitter_prediction=nt[body]['predicted_nt'],
            transmitter_confidence=nt[body]['predicted_nt_confidence'], **counts[body])
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), ct1=report['ct1'])))


if __name__ == '__main__':
    main()
