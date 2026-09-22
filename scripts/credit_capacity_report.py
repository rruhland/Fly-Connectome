"""Publish verified offline capacity results and a held-out comparison plot."""
import json
from pathlib import Path

import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fly_connectome.data import checksum


def main():
    source = Path('runs/credit-capacity-v1')
    results = json.loads((source/'results.json').read_text())
    for path, digest in results['artifacts'].items():
        assert checksum(path) == digest
    assert checksum(Path('scripts/credit_capacity.py')) == results['script_sha256']
    feasibility = json.loads((source/'feasibility.json').read_text())
    assert len(feasibility) == 6 and all(row['presolve'] is False for row in feasibility.values())
    assert feasibility['split-no-quiet-control']['status'] == 0
    assert feasibility['split-no-quiet-control']['constraint_violation'] <= 1e-6
    for row in feasibility['split-no-quiet-control']['training_scores'].values():
        assert min(row['on_anticipation'], row['off_anticipation']) >= .1-1e-6
    combined = dict(fits=results, feasibility=feasibility,
        software=dict(numpy=np.__version__, scipy=scipy.__version__),
        feasibility_script_sha256=checksum(Path('scripts/credit_feasibility.py')),
        regression_fixture_sha256=checksum(Path('tests/fixtures/credit_capacity_feasible.npz')))
    out = Path('docs/experiments')
    (out/'2026-09-21-credit-capacity-results.json').write_text(json.dumps(combined, indent=2, allow_nan=False)+'\n')
    labels = ['Recorded', 'Shared\nMSE', 'Split\nMSE', 'Shared\nbalanced', 'Split\nbalanced']
    names = ['recorded', 'shared-mse', 'split-mse', 'shared-balanced', 'split-balanced']
    metrics = [results['variants'][name]['scores']['test3'] for name in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for offset, polarity, color in [(-.18, 'on', '#0072B2'), (.18, 'off', '#D55E00')]:
        axes[0].bar(x+offset, [m[polarity+'_anticipation'] for m in metrics], width=.34,
                    label=polarity.upper(), color=color)
    axes[0].axhline(.1, color='black', linestyle='--', linewidth=1)
    axes[0].set(ylabel='Mean signed event anticipation', title='Stronger forecasts with context split')
    axes[0].legend(frameon=False)
    axes[1].bar(x, [100*m['false_alarm_fraction'] for m in metrics], color='#666666')
    axes[1].axhline(5, color='black', linestyle='--', linewidth=1)
    axes[1].set(ylabel='Quiet false alarms (%)', title='Quiet selectivity remains insufficient')
    for ax in axes:
        ax.set_xticks(x, labels);ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Offline frozen-history capacity: held-out tempo 3; no neural learning')
    fig.savefig(out/'assets/2026-09-21-credit-capacity.png', dpi=180)
    print('Verified and published capacity fits, corrected feasibility results and plot.')


if __name__ == '__main__':
    main()
