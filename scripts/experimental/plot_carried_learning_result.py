"""Plot recorded live behavior; no fitting, selection or environment execution."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot(root, output):
    root = Path(root)
    spec = json.loads((root / 'spec.json').read_text())
    n = spec['games_per_phase']
    streams = [name for name, config in spec['streams'].items() if config['role'] == 'live']
    fig, axes = plt.subplots(len(streams), 1, figsize=(11, 7), sharex=True, sharey=True, squeeze=False)
    for axis, name in zip(axes[:, 0], streams):
        rows = [json.loads(p.read_text()) for p in sorted((root / name).glob('game-[0-9][0-9][0-9].json'))]
        xs, returns, firsts = [], [], []
        for i in range(7, len(rows)):
            events = [event for row in rows[i-7:i+1] for event in row['events']]
            first = [event for event in events if event['first']]
            xs.append(i + 1)
            returns.append(sum(e['hit'] for e in events) / len(events) if events else float('nan'))
            firsts.append(sum(e['hit'] for e in first) / len(first) if first else float('nan'))
        for phase in range(3):
            axis.axvspan((phase+1)*n-31.5, (phase+1)*n+.5, color='#dceef7', alpha=.6)
            selected = rows[(phase+1)*n-32:(phase+1)*n]
            events = [e for row in selected for e in row['events']]
            axis.text(phase*n+n*.73, 1.065, f"Final 32: {sum(e['hit'] for e in events)}/{len(events)}",
                      ha='center', fontsize=9, color='#20536b')
        axis.plot(xs, returns, color='#167399', linewidth=2.2, label='All returns')
        axis.plot(xs, firsts, color='#df7a31', linewidth=1.6, label='First returns')
        for boundary in (n, 2*n):
            axis.axvline(boundary+.5, color='#555555', linestyle='--', linewidth=1)
        axis.axhline(.95, color='#888888', linestyle=':', linewidth=1)
        axis.set(ylim=(-.03, 1.12), xlim=(.5, 3*n+.5), ylabel='Return rate', title=name.replace('-live', '').replace('replica', 'Independent learner '))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis='y', alpha=.15)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(loc='lower right', frameon=False, ncol=2)
    axes[-1, 0].set_xlabel('Chronological game — normal controls → reversed controls → normal controls')
    fig.suptitle('Carried online learning and adaptation', fontsize=17, y=.99)
    fig.text(.5, .94, 'Learner state carries across both hidden changes. Every exploratory game is included.', ha='center', fontsize=10)
    fig.text(.5, .015, 'Lines: rolling 8-game rates (descriptive). Blue bands: fixed final 32-game acceptance windows. Dotted line: 95%.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .91))
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('output')
    args = parser.parse_args()
    plot(args.directory, args.output)
