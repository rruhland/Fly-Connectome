"""Read-only local voltage modulation in anatomically annotated silent afferents."""
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS
from motion_stage_recovery import annotated_columns
from fly_connectome.data import checksum
from fly_connectome.sensor import Retina


def main():
    source_sha = checksum(SOURCE)
    m = torch.load(SOURCE, weights_only=True)['metadata']
    retina = Retina(**m['retina'])
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns = annotated_columns(m, retina, table)
    types = np.asarray(m['retina']['cell_types'])
    saved = torch.load('runs/motion-stage-audit-v1/bar-per-neuron-responses.pt',
                       weights_only=True)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  quantity='mean membrane voltage over 96 transit ticks, stimulus minus matched blank',
                  groups={})
    for center in (18, 46):
        region = (x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23)
        bins = retina.pixel_bins[region.flatten()].unique().numpy()
        row = report['groups'][str(center)] = {}
        for label in ('Mi4', 'Tm4', 'Tm9'):
            local = (types == label) & np.isin(columns, bins)
            cell = row[label] = dict(neurons=int(local.sum()))
            for polarity in ('on', 'off'):
                stimulus = saved['responses'][f'{center}-{polarity}-+1']['mean_voltage'].numpy()
                blank = saved['baselines'][polarity]['mean_voltage'].numpy()
                difference = stimulus[local]-blank[local]
                cell[polarity] = dict(mean=float(difference.mean()),
                    mean_absolute=float(np.abs(difference).mean()),
                    cells_above_001=int((np.abs(difference) > .001).sum()),
                    positive_cells=int((difference > 0).sum()),
                    negative_cells=int((difference < 0).sum()))
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    output = Path('runs/motion-stage-recovery-v1/subthreshold.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), groups=report['groups'])), flush=True)


if __name__ == '__main__':
    main()
