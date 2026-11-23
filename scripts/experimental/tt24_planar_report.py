"""Summarize paired planar learning curves and draw the static evidence figure."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_source import MODES


ROOT = Path('artifacts/tt24')
NAMES = {'lateral': 'Lateral motion', 'energy': 'Two-axis motion', 'damped': 'Powered rebounds'}
LABELS = {'pong': 'Native Pong', 'cold': 'Cold', 'shuffled': 'Shuffled source', 'hold': 'Stationary'}
COLORS = {'pong': '#186b83', 'cold': '#6b7280', 'shuffled': '#9c5e39', 'hold': '#272b35'}


def curves(source, stage):
    with gzip.open(source, 'rt') as stream:
        record = json.load(stream)
    rows = [cohort['stages'][stage] for cohort in record['cohorts']]
    frames = sorted(rows[0]['modes']['pong']['held'], key=int)
    observations = {mode: np.array([[entry['modes'][mode]['held'][count]['mean'] for count in frames]
                                    for entry in rows]) for mode in MODES}
    observations['hold'] = np.repeat(np.array([[entry['hold']['mean']] for entry in rows]), len(frames), axis=1)
    rng = np.random.default_rng(3641091)
    draw = rng.integers(len(rows), size=(5000, len(rows)))
    averages = {mode: values.mean(axis=0).tolist() for mode, values in observations.items()}
    paired = {}
    for mode in ('cold', 'shuffled', 'hold'):
        diff = observations['pong'] - observations[mode]
        bounds = np.quantile(diff[draw].mean(axis=1), (.025, .975), axis=0).T
        paired[mode] = dict(delta=diff.mean(axis=0).tolist(), interval=bounds.tolist(),
                            wins=(diff > 0).sum(axis=0).tolist(), losses=(diff < 0).sum(axis=0).tolist())
    peaks = {mode: max(entry['modes'][mode]['held'][frames[-1]]['peak'] for entry in rows) for mode in MODES}
    result = dict(checkpoints=[int(i) for i in frames], average=averages, paired=paired, peaks=peaks,
                  cohorts=len(rows), held_per_cohort=len(rows[0]['held_seeds']))
    if stage != 'lateral':
        result['fixed_vertical'] = {mode: float(np.mean([entry['modes'][mode]['fixed_vertical']['mean'] for entry in rows]))
                                    for mode in MODES}
        result['powered'] = {mode: float(np.mean([entry['modes'][mode]['held'][frames[-1]]['powered'] for entry in rows]))
                             for mode in MODES}
    return result


def figure(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.right': False,
                         'axes.spines.top': False, 'axes.edgecolor': '#b3b9bf', 'text.color': '#172b34',
                         'axes.labelcolor': '#172b34', 'xtick.color': '#465765', 'ytick.color': '#465765'})
    chart, axes = plt.subplots(ncols=len(data), figsize=(11.7, 4.35), sharey=True)
    for i, (stage, series) in enumerate(data.items()):
        axis = axes[i]
        for mode in (*MODES, 'hold'):
            axis.plot(series['checkpoints'], series['average'][mode], color=COLORS[mode],
                      marker=None if mode == 'hold' else 'o', markersize=4,
                      linestyle='--' if mode == 'hold' else '-', linewidth=2.3 if mode == 'pong' else 1.5,
                      label=LABELS[mode])
        axis.set_title(NAMES[stage], loc='left', fontsize=12.5, fontweight='bold')
        axis.set_xlabel('Practice drops')
        axis.set_xticks(series['checkpoints'])
        axis.set_ylim(-.04, 2.05)
        axis.grid(axis='y', alpha=.19)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('Mean verified rebounds')
    chart.suptitle('Native Pong credit in planar gravity', x=.105, y=.97, ha='left', fontsize=21, fontweight='bold')
    chart.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='upper left', bbox_to_anchor=(.094, .893),
                 frameon=False, columnspacing=1.5, handlelength=1.55)
    chart.subplots_adjust(left=.105, right=.982, top=.732, bottom=.15, wspace=.23)
    chart.savefig(output, dpi=180, facecolor='white')
    plt.close(chart)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=ROOT / 'planar-planner.json.gz')
    parser.add_argument('--powered', type=Path, default=ROOT / 'planar-powered.json.gz')
    parser.add_argument('--output', type=Path, default=ROOT / 'planar-summary.json')
    parser.add_argument('--figure', type=Path, default=ROOT / 'planar-learning.png')
    args = parser.parse_args()
    data = {stage: curves(args.main if stage != 'damped' else args.powered, stage) for stage in NAMES}
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    figure(data, args.figure)


if __name__ == '__main__':
    main()
