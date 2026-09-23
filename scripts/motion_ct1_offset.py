"""Static CT1 input-to-T5 hex-offset comparison with cell-level bootstrap."""

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy.spatial import cKDTree
import torch

from fly_connectome.data import checksum
from fly_connectome.sensor import Retina
from motion_ct1_synapse_geometry import CACHE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns


OUT = Path('docs/experiments/2026-09-23-ct1-offset-results.json')


def cell_means(body, values):
    ids, positions = np.unique(body, return_inverse=True)
    mean = (np.bincount(positions, weights=values)
            /np.bincount(positions))
    return ids, mean


def summarize(body, dq, dr):
    ids, per_cell = cell_means(body, dr)
    rounded = np.rint(dr).astype(np.int64)
    return dict(contacts=len(dr), cells=len(ids),
        contact_mean_dr=float(dr.mean()), contact_median_dr=float(np.median(dr)),
        contact_mean_dq=float(dq.mean()),
        cell_mean_dr=float(per_cell.mean()),
        cell_median_dr=float(np.median(per_cell)),
        cell_positive_fraction=float((per_cell > 0).mean()),
        dr_histogram={str(k): int((rounded == k).sum())
                      for k in range(-5, 6)},
        dr_below_minus5=int((rounded < -5).sum()),
        dr_above_plus5=int((rounded > 5).sum()))


def main():
    if not CACHE.exists():
        raise FileNotFoundError('run motion_ct1_synapse_geometry.py first')
    source_sha = checksum(SOURCE)
    metadata = torch.load(SOURCE, weights_only=True)['metadata']
    retina = Retina(**metadata['retina'])
    ids = np.asarray(metadata['graph']['body_ids'])
    types = np.asarray(metadata['retina']['cell_types'])
    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    source_columns = annotated_columns(metadata, retina, annotations,
                                       classes=('Tm1', 'Tm9'))
    target_columns, _ = infer_columns(metadata, retina, annotations)
    source_by_body = {int(ids[i]): int(source_columns[i])
                      for i in np.flatnonzero(np.isin(types, ('Tm1', 'Tm9')))}
    target_by_body = {int(ids[i]): int(target_columns[i])
                      for i in np.flatnonzero(np.isin(types, ('T5c', 'T5d')))}
    subtype_by_body = {int(ids[i]): types[i]
                       for i in np.flatnonzero(np.isin(types, ('T5c', 'T5d')))}
    with np.load(CACHE) as saved:
        input_body = saved['input_body']
        input_xyz = saved['input_xyz']
        output_body = saved['output_body']
        output_xyz = saved['output_xyz']
    if len(input_body) != 32728 or len(output_body) != 37186:
        raise AssertionError('CT1 synapse cache differs from raw contact counts')
    distance, nearest = cKDTree(input_xyz).query(output_xyz, k=1)
    input_bin = np.asarray([source_by_body[int(body)]
                            for body in input_body[nearest]])
    output_bin = np.asarray([target_by_body[int(body)]
                             for body in output_body])
    if (input_bin < 0).any() or (output_bin < 0).any():
        raise AssertionError('nearest CT1 partner lacks an optic column')
    hexes = np.asarray(retina.spec['hex_columns'])
    difference = hexes[input_bin]-hexes[output_bin]
    dq, dr = difference[:, 0], difference[:, 1]
    pixel_bins = retina.pixel_bins.numpy()
    pixel_y = np.repeat(np.arange(32), 64)
    pixels_per_bin = np.bincount(pixel_bins, minlength=len(hexes))
    mean_y = (np.bincount(pixel_bins, weights=pixel_y, minlength=len(hexes))
              /np.maximum(pixels_per_bin, 1))
    represented = pixels_per_bin > 0
    orientation = np.linalg.lstsq(
        np.column_stack((hexes[represented],
                         np.ones(int(represented.sum())))),
        mean_y[represented], rcond=None)[0]
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    fields = {}
    for center in (10, 22, 16):
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        fields[f'y{center}'] = retina.pixel_bins[field.flatten()].unique().numpy()
    subtype = np.asarray([subtype_by_body[int(body)] for body in output_body])
    report = dict(source_sha256=source_sha,
        annotations_sha256=checksum(ANNOTATIONS), cache_sha256=checksum(CACHE),
        nearest_site_k=1, near_radius_voxels=125,
        retina_pixel_y_per_hex_q=float(orientation[0]),
        retina_pixel_y_per_hex_r=float(orientation[1]),
        bootstrap_seed=20260923, bootstrap_samples=1000,
        conditions={}, passes_subtype_opposed_offset=False)
    rng = np.random.default_rng(20260923)
    for field, bins in fields.items():
        field_rows = report['conditions'][field] = {}
        for radius, near in (('all', np.ones(len(output_body), dtype=bool)),
                             ('within_125_voxels', distance <= 125)):
            rows = field_rows[radius] = {}
            means = {}
            for name in ('T5c', 'T5d'):
                keep = near & (subtype == name) & np.isin(output_bin, bins)
                if not keep.any():
                    raise AssertionError('empty CT1 subtype/field offset')
                rows[name] = summarize(output_body[keep], dq[keep], dr[keep])
                _, means[name] = cell_means(output_body[keep], dr[keep])
            c, d = means['T5c'], means['T5d']
            bootstrap = (rng.choice(c, size=(1000, len(c)), replace=True).mean(1)
                         -rng.choice(d, size=(1000, len(d)), replace=True).mean(1))
            interval = np.quantile(bootstrap, (.025, .975))
            difference = float(c.mean()-d.mean())
            rows['subtype_difference'] = dict(
                t5c_minus_t5d_cell_mean_dr=difference,
                bootstrap_p025=float(interval[0]),
                bootstrap_p975=float(interval[1]),
                opposed=bool(c.mean()*d.mean() < 0),
                nonzero_interval=bool(interval[0]*interval[1] > 0))
            print(json.dumps(dict(field=field, radius=radius,
                t5c_mean=float(c.mean()), t5d_mean=float(d.mean()),
                difference=rows['subtype_difference'])), flush=True)
    checks = [report['conditions'][field][radius]['subtype_difference']
              for field in ('y10', 'y22', 'y16')
              for radius in ('all', 'within_125_voxels')]
    signs = np.sign([row['t5c_minus_t5d_cell_mean_dr'] for row in checks])
    report['passes_subtype_opposed_offset'] = bool(
        np.all(signs == signs[0]) and signs[0] != 0
        and all(row['opposed'] and row['nonzero_interval'] for row in checks))
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passes_subtype_opposed_offset=report['passes_subtype_opposed_offset'])),
        flush=True)


if __name__ == '__main__':
    main()
