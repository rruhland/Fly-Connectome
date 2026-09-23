"""Read-only measured receptive-field offsets for T4/T5 input classes."""
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS
from fly_connectome.data import checksum


def main():
    source_sha = checksum(SOURCE)
    m = torch.load(SOURCE, weights_only=True)['metadata']
    g = m['graph']
    types = np.asarray(m['retina']['cell_types'])
    ids = np.asarray(g['body_ids'])
    pre, post, contacts = (np.asarray(g[k]) for k in ('pre', 'post', 'contacts'))
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    body, q, r = (table[k].to_numpy(zero_copy_only=False)
                  for k in ('bodyId', 'assignedOlHex1', 'assignedOlHex2'))
    locations = {int(b): (float(x), float(y)) for b, x, y in zip(body, q, r)
                 if np.isfinite(x) and np.isfinite(y)}
    positions = np.asarray([locations.get(int(b), (np.nan, np.nan)) for b in ids])
    result = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  reference='contact-weighted direct Mi1 center for T4 or Tm1/Tm2 center for T5',
                  offsets='presynaptic assignedOlHex1/2 minus inferred postsynaptic center',
                  groups={})
    for family, reference, sources in (
            ('T4', ('Mi1',), ('Mi1', 'Tm3', 'Mi9', 'Mi4')),
            ('T5', ('Tm1', 'Tm2'), ('Tm1', 'Tm2', 'Tm4', 'Tm9'))):
        for subtype in ('a', 'b', 'c', 'd'):
            label = family+subtype
            in_target = types[post] == label
            in_reference = in_target & np.isin(types[pre], reference)
            valid = in_reference & np.isfinite(positions[pre]).all(1)
            denominator = np.bincount(post[valid], weights=contacts[valid],
                                      minlength=len(ids))
            centers = np.column_stack([
                np.bincount(post[valid], weights=contacts[valid]*positions[pre[valid], axis],
                            minlength=len(ids))/np.maximum(denominator, 1)
                for axis in (0, 1)])
            row = dict(neurons=int((types == label).sum()), sources={})
            for source in sources:
                anatomical = in_target & (types[pre] == source)
                selected = (anatomical
                            & np.isfinite(positions[pre]).all(1)
                            & (denominator[post] > 0))
                offsets = positions[pre[selected]]-centers[post[selected]]
                mass = contacts[selected]
                if not len(mass) or not mass.sum():
                    row['sources'][source] = dict(anatomical_edges=int(anatomical.sum()),
                        located_edges=0, contacts=0,
                        mean_axial_offset=None, mean_xy_offset=None)
                    continue
                mean = np.average(offsets, weights=mass, axis=0)
                row['sources'][source] = dict(anatomical_edges=int(anatomical.sum()),
                    located_edges=int(selected.sum()),
                    contacts=int(mass.sum()),
                    mean_axial_offset=[float(mean[0]), float(mean[1])],
                    mean_xy_offset=[float(mean[0]+mean[1]/2),
                                    float(mean[1]*np.sqrt(3)/2)])
            result['groups'][label] = row
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    output = Path('runs/motion-stage-recovery-v1/geometry.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), groups=result['groups'])), flush=True)


if __name__ == '__main__':
    main()
