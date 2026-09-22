"""Plot saved continuous timing results without rerunning or fitting the model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fly_connectome.data import checksum


def main():
    report = json.loads(Path('docs/experiments/2026-09-22-continuous-timing-results.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(top=.84, bottom=.08, left=.07, right=.99, hspace=.4, wspace=.2)
    for row, condition in enumerate(['standard', 'omitted']):
        path = Path(f'runs/continuous-timing-v1/{condition}.npz')
        assert checksum(path) == report['artifacts'][str(path)]
        data = np.load(path)
        frames = np.arange(1, len(data['events']))
        for col, ax in enumerate(axes[row]):
            ax.plot(frames, data['amplitude'][:-1], color='0.75', lw=1, label='Ungated saved readout')
            ax.plot(frames, data['gated'][:-1], color='#1765a0', lw=1, label='Causal gate applied')
            event = data['events'][1:] != 0
            ax.scatter(frames[event], data['events'][1:][event], marker='x', s=18,
                       color='black', label='Observed event', zorder=3)
            ax.axhline(0, color='0.8', lw=.5)
            ax.set(ylim=(-1.12, 1.12), xlabel='Target frame (forecast issued one frame earlier)',
                   ylabel='Signed amplitude', title=f'{condition.capitalize()}: '+('full stream' if col == 0 else 'omission / recovery window'))
            if col == 0:
                ax.set_xlim(0, frames[-1])
                for start, interval in zip(data['scheduled_frames'][::24], [2, 3, 6, 4]):
                    ax.axvline(start, color='0.4', lw=.5, linestyle=':')
                    ax.text(start+2, -.9, f'{interval}f', fontsize=9)
            else:
                ax.set_xlim(198, 252)
                for frame in data['scheduled_frames'][60:62]:
                    ax.axvline(frame, color='#c55b14', linestyle=':', lw=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle('Frozen neural histories; unchanged offline amplitudes and causal timing reference', y=.98)
    fig.savefig('docs/experiments/assets/2026-09-22-continuous-timing.png', dpi=180, bbox_inches='tight')


if __name__ == '__main__':
    main()
