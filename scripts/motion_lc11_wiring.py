"""Read-only accounting of measured LC11 inputs omitted from the M1 roster."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import torch

from fly_connectome.data import checksum
from fly_connectome.extraction import _sign

from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS


WEIGHTS = Path('data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather')
TRANSMITTERS = Path('data/raw/body-neurotransmitters-male-cns-v1.0.feather')
OUT = Path('docs/experiments/2026-09-23-lc11-wiring-results.json')
SOURCE_TYPES = ('T2', 'T2a', 'T3')


def collect_target_edges(path, target_ids):
    """Read only IPC batches containing a target; never materialize the full edge table."""
    segments = [[], [], []]
    with pa.memory_map(str(path), 'r') as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            post = batch.column(1).to_numpy(zero_copy_only=False)
            mask = np.isin(post, target_ids)
            if not mask.any():
                continue
            for column, output in zip(range(3), segments):
                output.append(batch.column(column).to_numpy(zero_copy_only=False)[mask])
    return tuple(np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
                 for parts in segments)


def class_summary(pre, post, contacts, target_ids, source_ids):
    mask = np.isin(pre, source_ids)
    source_pre, source_post, source_contacts = pre[mask], post[mask], contacts[mask]
    totals = [int(source_contacts[source_post == target].sum()) for target in target_ids]
    reached = [total for total in totals if total > 0]
    return dict(source_neurons=int(len(source_ids)),
                source_neurons_connected=int(len(np.unique(source_pre))),
                edges=int(mask.sum()),
                edges_at_least_3=int((source_contacts >= 3).sum()),
                contacts=int(source_contacts.sum()),
                targets_reached=len(reached),
                target_fraction=len(reached)/len(target_ids),
                median_contacts_per_reached_target=(float(np.median(reached))
                                                    if reached else 0.))


def summarize_side(annotations, selected_ids, pre, post, contacts, side):
    targets = np.array(sorted(body for body, row in annotations.items()
                              if row['type'] == 'LC11' and row['somaSide'] == side
                              and row['status'] == 'Traced'), dtype=np.int64)
    keep = np.isin(post, targets)
    pre, post, contacts = pre[keep], post[keep], contacts[keep]
    selected = (class_summary(pre, post, contacts, targets, selected_ids)
                if side == 'R' else None)
    classes = {}
    for cell_type in SOURCE_TYPES:
        source_ids = np.array([body for body, row in annotations.items()
                               if row['type'] == cell_type and row['somaSide'] == side
                               and row['status'] == 'Traced'],
                              dtype=np.int64)
        classes[cell_type] = class_summary(pre, post, contacts, targets, source_ids)
    totals = {}
    for body, weight in zip(pre, contacts):
        cell_type = annotations.get(int(body), {}).get('type') or 'untyped'
        totals[cell_type] = totals.get(cell_type, 0) + int(weight)
    top = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:10]
    return dict(target_neurons=len(targets), total_edges=len(pre),
                total_contacts=int(contacts.sum()),
                selected_input=selected, same_side_source_classes=classes,
                largest_source_classes=[dict(cell_type=name, contacts=mass)
                                        for name, mass in top])


def source_signs(annotations, transmitter_rows, selected_ids):
    nt = {int(row['body']): row for row in transmitter_rows}
    result = {}
    for cell_type in SOURCE_TYPES:
        counts = {'excitatory': 0, 'inhibitory': 0, 'unresolved': 0}
        for body in selected_ids:
            if annotations.get(int(body), {}).get('type') != cell_type:
                continue
            sign = _sign(nt.get(int(body), {}), .5)
            key = 'unresolved' if sign is None else ('excitatory' if sign[0] > 0
                                                     else 'inhibitory')
            counts[key] += 1
        result[cell_type] = counts
    return result


def main():
    payload = torch.load(SOURCE, weights_only=True)
    selected = np.asarray(payload['metadata']['graph']['body_ids'], dtype=np.int64)
    columns = ['bodyId', 'type', 'somaSide', 'status']
    table = feather.read_table(ANNOTATIONS, columns=columns)
    rows = table.to_pylist()
    annotations = {int(row['bodyId']): row for row in rows}
    targets = np.array([body for body, row in annotations.items()
                        if row['type'] == 'LC11' and row['status'] == 'Traced'],
                       dtype=np.int64)
    pre, post, contacts = collect_target_edges(WEIGHTS, targets)
    transmitter_rows = feather.read_table(TRANSMITTERS).to_pylist()
    report = dict(source_sha256=checksum(SOURCE),
                  annotations_sha256=checksum(ANNOTATIONS),
                  weights_sha256=checksum(WEIGHTS),
                  transmitters_sha256=checksum(TRANSMITTERS),
                  source_signs=source_signs(annotations, transmitter_rows, selected),
                  right=summarize_side(annotations, selected, pre, post, contacts, 'R'),
                  left=summarize_side(annotations, selected, pre, post, contacts, 'L'))
    OUT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
