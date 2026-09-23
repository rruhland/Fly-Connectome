"""Read-only CT1 SWC cable-nearest synapse assignment from MaleCNS."""

import heapq
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pyarrow.feather as feather
from scipy.spatial import cKDTree
import torch

from fly_connectome.data import checksum
from fly_connectome.sensor import Retina
from motion_ct1_offset import cell_means
from motion_ct1_synapse_geometry import CACHE as SITES
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns


SKELETON_URL = ('https://storage.googleapis.com/flyem-male-cns/v1.0/'
    'segmentation/skeletons-malecns/skeletons-swc/10009.swc')
SKELETON = Path('runs/motion-ct1-cable-v1/10009.swc')
OUT = Path('docs/experiments/2026-09-23-ct1-cable-results.json')


def cable_nearest(node_xyz, parent_ids, node_ids, input_nodes,
                  input_attachment):
    """Multi-source shortest cable path to an indexed input site."""
    positions = {int(node): i for i, node in enumerate(node_ids)}
    adjacency = [[] for _ in node_ids]
    roots = 0
    for child, parent in enumerate(parent_ids):
        if parent == -1:
            roots += 1
            continue
        if int(parent) not in positions:
            raise AssertionError('SWC parent absent from skeleton')
        other = positions[int(parent)]
        length = float(np.linalg.norm(node_xyz[child]-node_xyz[other]))
        adjacency[child].append((other, length))
        adjacency[other].append((child, length))
    if roots < 1:
        raise AssertionError('SWC skeleton has no root')
    distance = np.full(len(node_ids), np.inf)
    nearest = np.full(len(node_ids), -1, dtype=np.int64)
    heap = []
    for site, (node, attachment) in enumerate(zip(input_nodes, input_attachment)):
        if attachment < distance[node]:
            distance[node] = attachment
            nearest[node] = site
            heapq.heappush(heap, (float(attachment), int(node)))
    while heap:
        value, node = heapq.heappop(heap)
        if value != distance[node]:
            continue
        for neighbor, length in adjacency[node]:
            proposal = value+length
            if proposal < distance[neighbor]:
                distance[neighbor] = proposal
                nearest[neighbor] = nearest[node]
                heapq.heappush(heap, (proposal, neighbor))
    return distance, nearest, roots


def main():
    if not SITES.exists():
        raise FileNotFoundError('run motion_ct1_synapse_geometry.py first')
    if not SKELETON.exists():
        SKELETON.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(SKELETON_URL, SKELETON)
    swc = np.loadtxt(SKELETON, comments='#')
    if swc.ndim != 2 or swc.shape[1] < 7:
        raise AssertionError('unexpected CT1 SWC columns')
    node_ids = swc[:, 0].astype(np.int64)
    node_xyz = swc[:, 2:5]
    parents = swc[:, 6].astype(np.int64)
    if len(np.unique(node_ids)) != len(node_ids):
        raise AssertionError('duplicate CT1 skeleton node')
    with np.load(SITES) as saved:
        input_body = saved['input_body']
        input_xyz = saved['input_xyz']
        output_body = saved['output_body']
        output_xyz = saved['output_xyz']
    if len(input_body) != 32728 or len(output_body) != 37186:
        raise AssertionError('CT1 pair counts differ from raw connectome')
    tree = cKDTree(node_xyz)
    input_attachment, input_nodes = tree.query(input_xyz, k=1)
    output_attachment, output_nodes = tree.query(output_xyz, k=1)
    attachment_median = max(float(np.median(input_attachment)),
                            float(np.median(output_attachment)))
    if attachment_median > 125:
        report = dict(source_sha256=checksum(SOURCE),
            sites_sha256=checksum(SITES), skeleton_url=SKELETON_URL,
            skeleton_sha256=checksum(SKELETON), coordinate_units='8 nm voxels',
            skeleton_nodes=len(node_ids), input_sites=len(input_body),
            output_sites=len(output_body), maximum_median_attachment_voxels=125,
            attachment_distance_voxels=dict(
                input_median=float(np.median(input_attachment)),
                input_p95=float(np.quantile(input_attachment, .95)),
                output_median=float(np.median(output_attachment)),
                output_p95=float(np.quantile(output_attachment, .95))),
            passes_attachment_screen=False,
            stopped_before_cable_assignment=True)
        OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(OUT),
            stopped_before_cable_assignment=True,
            attachment_distance_voxels=report['attachment_distance_voxels'])),
            flush=True)
        return
    cable_distance, labels, roots = cable_nearest(
        node_xyz, parents, node_ids, input_nodes, input_attachment)
    cable_source = labels[output_nodes]
    if (cable_source < 0).any():
        raise AssertionError('CT1 output branch has no input cable path')
    output_cable_distance = cable_distance[output_nodes]+output_attachment
    euclidean_distance, euclidean_source = cKDTree(input_xyz).query(output_xyz,
                                                                     k=1)
    metadata = torch.load(SOURCE, weights_only=True)['metadata']
    retina = Retina(**metadata['retina'])
    ids = np.asarray(metadata['graph']['body_ids'])
    types = np.asarray(metadata['retina']['cell_types'])
    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    source_column = annotated_columns(metadata, retina, annotations,
                                      classes=('Tm1', 'Tm9'))
    target_column, _ = infer_columns(metadata, retina, annotations)
    source_by_body = {int(ids[i]): int(source_column[i])
                      for i in np.flatnonzero(np.isin(types, ('Tm1', 'Tm9')))}
    target_by_body = {int(ids[i]): int(target_column[i])
                      for i in np.flatnonzero(np.isin(types, ('T5c', 'T5d')))}
    type_by_body = {int(ids[i]): types[i]
                    for i in np.flatnonzero(np.isin(types, ('T5c', 'T5d')))}
    hexes = np.asarray(retina.spec['hex_columns'])
    output_bin = np.asarray([target_by_body[int(body)] for body in output_body])
    cable_bin = np.asarray([source_by_body[int(body)]
                            for body in input_body[cable_source]])
    euclidean_bin = np.asarray([source_by_body[int(body)]
                                for body in input_body[euclidean_source]])
    if min(output_bin.min(), cable_bin.min(), euclidean_bin.min()) < 0:
        raise AssertionError('CT1 partner lacks an optic column')
    cable_dr = hexes[cable_bin, 1]-hexes[output_bin, 1]
    euclidean_dr = hexes[euclidean_bin, 1]-hexes[output_bin, 1]
    subtype = np.asarray([type_by_body[int(body)] for body in output_body])
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    fields = {}
    for center in (10, 22, 16):
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        fields[f'y{center}'] = retina.pixel_bins[field.flatten()].unique().numpy()
    rng = np.random.default_rng(20260923)
    report = dict(source_sha256=checksum(SOURCE),
        annotations_sha256=checksum(ANNOTATIONS),
        sites_sha256=checksum(SITES), skeleton_url=SKELETON_URL,
        skeleton_sha256=checksum(SKELETON), coordinate_units='8 nm voxels',
        skeleton_nodes=len(node_ids), skeleton_roots=roots,
        input_sites=len(input_body), output_sites=len(output_body),
        attachment_distance_voxels=dict(
            input_median=float(np.median(input_attachment)),
            input_p95=float(np.quantile(input_attachment, .95)),
            output_median=float(np.median(output_attachment)),
            output_p95=float(np.quantile(output_attachment, .95))),
        cable_distance_voxels=dict(median=float(np.median(output_cable_distance)),
            p95=float(np.quantile(output_cable_distance, .95))),
        euclidean_distance_voxels=dict(median=float(np.median(euclidean_distance)),
            p95=float(np.quantile(euclidean_distance, .95))),
        changed_source_body_fraction=float((input_body[cable_source]
            != input_body[euclidean_source]).mean()),
        changed_source_column_fraction=float((cable_bin != euclidean_bin).mean()),
        bootstrap_seed=20260923, bootstrap_samples=1000,
        fields={}, passes_null_leading_cable_screen=False)
    checks = []
    for field, bins in fields.items():
        field_rows = report['fields'][field] = {}
        for name in ('T5c', 'T5d'):
            keep = (subtype == name) & np.isin(output_bin, bins)
            _, cell_offset = cell_means(output_body[keep], cable_dr[keep])
            bootstrap = rng.choice(cell_offset,
                size=(1000, len(cell_offset)), replace=True).mean(1)
            interval = np.quantile(bootstrap, (.025, .975))
            mean = float(cell_offset.mean())
            pass_sign = bool(interval[1] < 0 if name == 'T5c'
                             else interval[0] > 0)
            checks.append(pass_sign)
            field_rows[name] = dict(contacts=int(keep.sum()),
                cells=len(cell_offset),
                cable_cell_mean_dr=mean,
                cable_cell_median_dr=float(np.median(cell_offset)),
                euclidean_contact_mean_dr=float(euclidean_dr[keep].mean()),
                cable_bootstrap_p025=float(interval[0]),
                cable_bootstrap_p975=float(interval[1]),
                null_leading_sign=pass_sign,
                changed_source_column_fraction=float((cable_bin[keep]
                    != euclidean_bin[keep]).mean()))
        print(json.dumps(dict(field=field, rows=field_rows)), flush=True)
    report['passes_null_leading_cable_screen'] = all(checks)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), skeleton_nodes=len(node_ids),
        passes_null_leading_cable_screen=report['passes_null_leading_cable_screen'])),
        flush=True)


if __name__ == '__main__':
    main()
