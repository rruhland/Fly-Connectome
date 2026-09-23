"""Read-only measured T5c/d output and predictive credit into LPi targets."""

import json
from pathlib import Path

import numpy as np
import torch

from fly_connectome.data import checksum
from fly_connectome.graph import Graph
from motion_stage_audit import SOURCE


OUT = Path('docs/experiments/2026-09-23-t5-downstream-wiring-results.json')
TARGETS = ('LPi34', 'LPi43', 'LPi3412')


def main():
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    types = np.asarray(metadata['retina']['cell_types'])
    pre, post, contacts = graph.pre, graph.post, graph.contacts
    pathways = np.asarray(metadata['pathways'])
    report = dict(source_sha256=checksum(SOURCE), graph_sha256=graph.identity(),
                  targets={})
    for target_type in TARGETS:
        cells = np.flatnonzero(types == target_type)
        feedforward = (pathways == 'feedforward') & np.isin(post, cells)
        predictive = (pathways == 'predictive') & np.isin(post, cells)
        row = dict(cells=len(cells), t5={},
            predictive_incoming_edges=int(predictive.sum()),
            predictive_incoming_contacts=int(contacts[predictive].sum()))
        for source_type in ('T5c', 'T5d'):
            selected = feedforward & (types[pre] == source_type)
            row['t5'][source_type] = dict(edges=int(selected.sum()),
                contacts=int(contacts[selected].sum()),
                fixed_positive_signs=bool((graph.signs[pre[selected]] == 1).all()))
        source_types, counts = np.unique(types[pre[predictive]], return_counts=True)
        order = np.argsort(-counts)[:8]
        row['top_predictive_sources'] = {
            str(source_types[i]): int(counts[i]) for i in order}
        report['targets'][target_type] = row
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), targets=report['targets'])))


if __name__ == '__main__':
    main()
